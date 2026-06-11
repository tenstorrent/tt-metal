# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TTNN LM head (untied vocab projection) for dots.ocr.

The dots.ocr text decoder ends in an untied ``Linear(hidden -> vocab)``,
1536 -> 151936, no bias (``lm_head.weight``, separate from
``model.embed_tokens.weight``). Reference: a single ``F.linear`` in
reference/functional.py ``lm_head_forward``; reference_impl
models/tt_transformers/tt/lm_head.py (its DRAM-sharded program configs,
column splitting against L1 OOM, and prefetcher/ring-matmul paths are
optimization-phase concerns — first-pass correctness uses one plain
``ttnn.linear`` per chip into DRAM).

Parallelism plan (ARCHITECTURE.md / inventory notes): placement=shard on
the VOCAB dim — the classic column-parallel LM head from tp-guidance:
weight ``[hidden, vocab]`` is sharded ``ShardTensorToMesh(dim=-1)`` so each
chip of the 1x4 mesh computes its contiguous 151936/4 = 37984-logit slice
(1187 tiles, no padding needed) from the replicated activation. No CCL in
the block: per the plan the logits are recombined by host concat
(``ConcatMeshToTensor(dim=-1)``) or an optional ``all_gather`` left to the
generation phase, where the consumer (sampling/argmax) decides what it
needs.

Occupancy redo (optimization phase, tick-66): the production decode call —
ONE bf16 tile row [1,1,1,1536] -> 37984 fp32 logits/chip per token — was
profiled under traced replay at the queried 11x10=110 grid. The single
matmul runs on 108/110 cores (98% occupancy) streaming the 62 MB bfp8
weight slice at ~329 GB/s: 188.4 us/device, weight-BW-bound (compute est
~27 us at HiFi4, 7x below the stream time). Levers measured and REVERTED
with evidence:

- DRAM-WIDTH-SHARDED weight + MatmulMultiCoreReuseMultiCastDRAMSharded
  (the text_mlp/text_attention decode recipe): 48 cores is the only grid
  that fits (hidden = 48 K-tiles caps the divisor; 24 cores TT_THROWs at
  1.72 MB static CBs > 1.46 MB L1 — per_core_N=50 fp32 interm/out CBs).
  Measured 419.7 us matmul (~148 GB/s, the few-core DRAM-sharded BW
  plateau scaled to 62 MB) + 25 us pad-slice + 16.6 us s2i = 466.7
  us/device vs 188.4 baseline. The interleaved 108-core reader is the
  better streamer at this weight size — LOSES 2.5x, reverted.
- bfloat4_b weight (halves streamed bytes): worst logits PCC 0.9884 <
  0.99 over 16 production rows and 6/16 greedy argmax flips vs the fp32
  reference — disqualified, bfp8 stays.
- HiFi2 waved off by arithmetic: the op is bytes-bound by 7x margin;
  fidelity changes FLOP passes, not bytes.

Verdict: at-ceiling-with-evidence at 98% occupancy; the plain
``ttnn.linear`` below IS the optimized form.
"""

import ttnn
from models.common.lightweightmodule import LightweightModule


class TtLMHead(LightweightModule):
    """dots.ocr LM head: x @ W^T, 1536 -> 151936, no bias, vocab-sharded.

    Args:
        mesh_device: ttnn mesh device handle (1xN line; weight vocab-sharded).
        state_dict: {"weight": [vocab=151936, hidden=1536]} torch tensor
            (HF key lm_head.weight, untied).
        dtype: on-device weight dtype. Default bfloat8_b: the 1536->37984
            per-chip matmul is DRAM weight-streaming-bound (~111MB bf16 at
            ~350us/device, tracy tick-31), so halving weight bytes halves the
            block; HiFi4+fp32-acc accumulation keeps logits PCC > 0.99
            (tt_transformers lm_head idiom).
    """

    def __init__(self, mesh_device, state_dict, dtype=ttnn.bfloat8_b):
        super().__init__()
        self.mesh_device = mesh_device
        num_devices = mesh_device.get_num_devices()
        self.num_devices = num_devices

        weight = state_dict["weight"]  # [vocab, hidden]
        self.vocab_size = weight.shape[0]
        assert (
            self.vocab_size % (32 * num_devices) == 0
        ), f"vocab {self.vocab_size} not tile-divisible across {num_devices} devices"

        # [vocab, hidden] -> [hidden, vocab] for x @ W^T, then column-parallel:
        # shard the OUTPUT (vocab) dim so each chip owns a contiguous logit slice.
        self.weight = ttnn.from_torch(
            weight.transpose(-2, -1).contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=(
                ttnn.ShardTensorToMesh(mesh_device, dim=-1)
                if num_devices > 1
                else ttnn.ReplicateTensorToMesh(mesh_device)
            ),
        )

        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x: [1, 1, seq, hidden] TILE_LAYOUT, replicated across the mesh.

        Returns: [1, 1, seq, vocab/N] per chip — logits sharded on the vocab
        dim, ALWAYS fp32 (greedy argmax near-tie exactness is independent of
        the weight storage dtype; HiFi4 + fp32 dest acc accumulate exactly).
        Recombine with ``ConcatMeshToTensor(dim=-1)`` on host (or an
        ``all_gather`` if a later consumer needs replicated logits on device).
        """
        return ttnn.linear(
            x,
            self.weight,
            dtype=ttnn.float32,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
