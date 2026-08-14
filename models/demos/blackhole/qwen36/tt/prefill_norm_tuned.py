# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DistributedNorm with a TUNED prefill all-gather.

WHY THIS FILE EXISTS
--------------------
``DistributedNorm.forward`` in ``models/tt_transformers/tt/distributed_norm.py`` reads
``chunks_per_sync`` / ``num_workers_per_link`` / ``num_links`` from ``model_config`` **only when
mode == "decode"**; every other mode gets the literals ``10`` / ``2``. So the per-op CCL tuning that
exists for decode has no prefill counterpart, and the prefill all-gather has never been tuned.

On an N300 that all-gather is one of the two largest ops in a GDN prefill layer. MEASURED at seq 2048
(single-layer profile, both all-gathers in the layer, 2 runs per config):

    num_workers_per_link=2 (upstream default)   1,245 + 1,242 us   = 2,487..2,524
    num_workers_per_link=4                      1,013 + 1,061 us   = 2,074..2,080
    num_workers_per_link=8                      1,020 + 1,016 us   = 2,028..2,036   <- used

~473us per layer, and extremely repeatable (8us spread across runs). ``chunks_per_sync`` made no
measurable difference and stays at the upstream default.

WHAT THIS OVERRIDES
-------------------
BOTH prefill all-gather branches of ``DistributedNorm.forward``, which upstream leaves untuned in
different places. Which branch a model takes is decided by ``is_distributed_norm(mode)``, and on a 1D
mesh that is ``dim > 4096 and mode == PREFILL`` -- so **the branch is the model**:

    9B  (dim 4096) -> is_distributed_norm False -> PRE-norm gather  (the original branch above)
    27B (dim 5120) -> is_distributed_norm True  -> POST-norm gather (added 2026-08-14)

The two are mutually exclusive by construction, which is what makes this file's tuning model-gated
without a single ``if dim`` test: the 9B can never reach the post-norm branch and the 27B can never
reach the pre-norm one. TG and decode still delegate to ``super().forward()``.

THE POST-NORM GATHER (27B)
--------------------------
Upstream's post-norm gather passes the literals ``chunks_per_sync=10`` / ``num_workers_per_link=2``
unconditionally -- it has no ``ag_config_key`` escape hatch at all, not even for decode. This gather
is the single largest op in the 27B's MLP block (~34% of it at seq 2048), and it exists only because
the fused AG+matmul (``tp_common.mlp_gateup_agmm_enabled``) is Blackhole-only -- see that function
for the C++ blocker that keeps it off on Wormhole.

**The worker tuning that works on the 9B does NOT transfer, and this is the useful finding.**
MEASURED (T3K, TP=8, 27B, single-layer GDN prefill at seq 2048, ff_norm's gather
[1,1,2048,640] -> [1,1,2048,5120] bf16, device kernel duration):

    num_workers_per_link=2 (upstream default)   1,201.6 us   (6 cores)
    num_workers_per_link=4                      1,198.7 us   (10 cores)
    num_workers_per_link=8                      1,163.7 us   (18 cores)

i.e. a wash -- -3% at 4x the workers, against ~100us of run-to-run spread on the MLP block. The
core count moves with the setting, so it IS taking effect; this gather simply is not worker-bound
the way the 9B's N300 gather was (-19% there). At TP=8 it is 7 ring hops of 2.62MB/device over
``get_num_links()==1`` ETH link, so **bytes are the only lever**, which is what
``prefill_gather_dtype`` below exists for: bf8 takes it 1,196.8 -> 678.2 us (-43%) for one 20.9us
typecast. This branch still reads ``prefill_ccl_tuning()`` (so there is one knob, not two) and so
runs at wpl=4 -- that value is the 9B's, chosen there on its own measurements, and is deliberately
NOT re-tuned for the 27B because at TP=8 it makes no difference either way.

The mirrored branches are a few lines of upstream logic each. ``_check_upstream_shape()`` asserts the
upstream branches still look the way these mirrors assume, so drift fails loudly rather than silently
running a stale duplicate.
"""
import inspect

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.distributed_norm import DistributedNorm

# The upstream literals this class exists to replace. If upstream starts honouring model_config for
# non-decode modes (or otherwise restructures the gather), this mirror is stale.
_UPSTREAM_ANCHORS = (
    "if self.args.is_multichip and not self.args.is_distributed_norm(mode):",
    "num_buffers_per_channel=2,",
    # the post-norm gather branch (the 27B's), mirrored below
    "if self.args.is_distributed_norm(mode) and self.enable_all_gather:",
)
_checked = False


def _check_upstream_shape():
    global _checked
    if _checked:
        return
    _checked = True
    try:
        src = inspect.getsource(DistributedNorm.forward)
    except (OSError, TypeError):  # source unavailable (zipimport etc.) -- skip
        return
    missing = [a for a in _UPSTREAM_ANCHORS if a not in src]
    if missing:
        raise RuntimeError(
            "models/demos/blackhole/qwen36/tt/prefill_norm_tuned.py mirrors the prefill all-gather "
            f"branches of DistributedNorm.forward, and upstream has changed: {missing!r} no longer "
            "present. Re-check those branches and re-apply the tuned chunks_per_sync / "
            "num_workers_per_link (see this module's docstring for the measurements)."
        )


class PrefillTunedDistributedNorm(DistributedNorm):
    """DistributedNorm whose PREFILL all-gather uses tp_common.prefill_ccl_tuning().

    Covers both gather positions: the pre-norm one (9B) and the post-norm one the distributed-norm
    path uses (27B). See the module docstring for why the branch choice *is* the model gate.
    """

    def __init__(self, *args, prefill_gather_dtype=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Narrow the POST-norm prefill gather to this dtype before sending it (see _forward_distributed).
        # None keeps the norm's own dtype, which is what every caller but ff_norm passes.
        self.prefill_gather_dtype = prefill_gather_dtype

    def forward(self, x, mode: Mode, norm_config=None):
        if mode == Mode.DECODE or self.TG or not self.args.is_multichip:
            return super().forward(x, mode, norm_config=norm_config)

        if self.args.is_distributed_norm(mode):
            return self._forward_distributed(x, mode, norm_config)

        _check_upstream_shape()
        chunks_per_sync, num_workers_per_link = tpc.prefill_ccl_tuning()
        # Mirrors upstream's non-decode branch exactly, except for the two tuned kwargs.
        x = ttnn.experimental.all_gather_async(
            x,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
            num_links=self.tt_ccl.get_num_links(1),
            topology=self.args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            chunks_per_sync=chunks_per_sync,
            num_workers_per_link=num_workers_per_link,
            num_buffers_per_channel=2,
            subdevice_id=self.prefetcher.worker_sub_device_id if self.prefetcher is not None else None,
        )
        # in/out_sharded are False for every non-DECODE mode upstream; the post-norm gather upstream
        # does is gated on is_distributed_norm(mode), which is False on this branch by construction.
        return self.norm(x, mode=mode, in_sharded=False, out_sharded=False, norm_config=norm_config)

    def _forward_distributed(self, x, mode: Mode, norm_config):
        """Mirror of upstream's is_distributed_norm branch: norm on the shard, then gather the result.

        Upstream reaches the same gather with hardcoded chunks_per_sync=10 / num_workers_per_link=2.
        The only departures here are those two kwargs.
        """
        _check_upstream_shape()
        chunks_per_sync, num_workers_per_link = tpc.prefill_ccl_tuning()

        # Upstream's `else` limb of the pre-norm test: no gather, just the memory config. For every
        # non-DECODE mode that is DRAM_MEMORY_CONFIG.
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        x = self.norm(x, mode=mode, in_sharded=False, out_sharded=False, norm_config=norm_config)
        if not self.enable_all_gather:
            return x
        if self.prefill_gather_dtype is not None and x.dtype != self.prefill_gather_dtype:
            # Narrow BEFORE the gather: this collective is bytes-bound on one ETH link, so the dtype
            # is the only lever that moves it (num_workers_per_link is a wash at TP=8 -- measured
            # 1201.6/1198.7/1163.7us for wpl 2/4/8). See the module docstring.
            narrowed = ttnn.typecast(x, self.prefill_gather_dtype)
            ttnn.deallocate(x)
            x = narrowed
        return ttnn.experimental.all_gather_async(
            x,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
            num_links=self.tt_ccl.get_num_links(1),
            topology=self.args.ccl_topology(),
            memory_config=x.memory_config(),
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            chunks_per_sync=chunks_per_sync,
            num_workers_per_link=num_workers_per_link,
            num_buffers_per_channel=2,
        )
