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
Only the branch we tune: multichip, non-TG, ``is_distributed_norm(mode)`` False -- which is exactly
the Wormhole prefill case (``is_distributed_norm`` returns False because ``dim == 4096`` is not
``> 4096``). Everything else -- TG, decode, and the distributed-norm path with its post-norm gather --
delegates to ``super().forward()``, so this file cannot change those.

The mirrored branch is 4 lines of upstream logic (all-gather with mode-appropriate memory config, then
the norm with in/out_sharded False for prefill). ``_check_upstream_shape()`` asserts the upstream
branch still looks the way this mirror assumes, so drift fails loudly rather than silently running a
stale duplicate.
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
            "models/demos/blackhole/qwen36/tt/prefill_norm_tuned.py mirrors the pre-norm all-gather "
            f"branch of DistributedNorm.forward, and upstream has changed: {missing!r} no longer "
            "present. Re-check that branch and re-apply the tuned chunks_per_sync / "
            "num_workers_per_link (see this module's docstring for the measurements)."
        )


class PrefillTunedDistributedNorm(DistributedNorm):
    """DistributedNorm whose PREFILL pre-norm all-gather uses tp_common.prefill_ccl_tuning()."""

    def forward(self, x, mode: Mode, norm_config=None):
        _tuned_branch = (
            mode != Mode.DECODE and not self.TG and self.args.is_multichip and not self.args.is_distributed_norm(mode)
        )
        if not _tuned_branch:
            return super().forward(x, mode, norm_config=norm_config)

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
