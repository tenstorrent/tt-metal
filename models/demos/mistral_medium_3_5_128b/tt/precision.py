# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness knobs (measurements, not spec dtypes). ``MISTRAL_PRECISION`` is a comma-separated list of
flags; unset = the package defaults below. Read once when a module is built.

  sdpa_bf16_acc     no-cache ring-joint SDPA with fp32_dest_acc_en=False (M3's setting; the default is fp32
                    accumulation; the cache-read ring path always runs without it, the op requires that)
  no_packer_l1_acc  projection matmuls without packer L1 accumulation
  residual_fp32     keep the residual stream (embedding, residual adds, norm inputs) in fp32
  proj_out_fp32     o_proj / down_proj emit fp32 partial sums into the TP reduce-scatter

``MISTRAL_SDPA_CHUNKS=q,k`` overrides the ring-joint SDPA q/k chunk sizes (default 128,512).

The measured effect of each on the full-depth KV PCC is in README.md ("Correctness knobs").
"""

import os
from dataclasses import dataclass

import ttnn

FLAGS = ("sdpa_bf16_acc", "no_packer_l1_acc", "residual_fp32", "proj_out_fp32")


@dataclass(frozen=True)
class Precision:
    sdpa_fp32_acc: bool = True
    packer_l1_acc: bool = True
    residual_fp32: bool = False
    proj_out_fp32: bool = False
    sdpa_q_chunk: int = 128
    sdpa_k_chunk: int = 512

    @property
    def residual_dtype(self):
        return ttnn.float32 if self.residual_fp32 else ttnn.bfloat16

    @property
    def proj_out_dtype(self):
        return ttnn.float32 if self.proj_out_fp32 else ttnn.bfloat16


def precision() -> Precision:
    flags = {f.strip() for f in os.environ.get("MISTRAL_PRECISION", "").split(",") if f.strip()}
    unknown = flags - set(FLAGS)
    if unknown:
        raise ValueError(f"MISTRAL_PRECISION: unknown flags {sorted(unknown)}; valid: {FLAGS}")
    q_chunk, k_chunk = (int(v) for v in os.environ.get("MISTRAL_SDPA_CHUNKS", "128,512").split(","))
    return Precision(
        sdpa_q_chunk=q_chunk,
        sdpa_k_chunk=k_chunk,
        sdpa_fp32_acc="sdpa_bf16_acc" not in flags,
        packer_l1_acc="no_packer_l1_acc" not in flags,
        residual_fp32="residual_fp32" in flags,
        proj_out_fp32="proj_out_fp32" in flags,
    )
