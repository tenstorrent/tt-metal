"""t457 noise-floor control: dense ring SDPA with k_chunk capped at 256 (default 512), same precision.

Only the k-chunk accumulation order changes (more running max/sum rescales, different bf16 rounding). PSNR of this
dense arm vs the stock dense arm is the trajectory floor for any equal-precision attention substitution.
Turbo 4x8 (SP 8) takes ring_joint_sdpa with `_sdpa_program_config(ring=True)`; the exp-ring path is off there
(job 041 patched only exp-ring and came out bit-identical).
"""
import os

import ttnn
from models.tt_dit.models.transformers.minimax_h3 import attention_minimax_h3 as _am

_K = int(os.environ.get("T457_KMAX", "256"))
_orig = _am.MiniMaxH3Attention._sdpa_program_config
_seen = set()


def _capped(self, seq_local, *, ring, windowed=False):
    pc = _orig(self, seq_local, ring=ring, windowed=windowed)
    if pc.k_chunk_size <= _K:
        return pc
    new = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=pc.compute_with_storage_grid_size,
        q_chunk_size=pc.q_chunk_size,
        k_chunk_size=_K,
        exp_approx_mode=False,
    )
    self._sdpa_program_configs[(seq_local, ring, windowed)] = new
    if (seq_local, ring) not in _seen:
        _seen.add((seq_local, ring))
        print(
            f"[t457] sdpa seq_local {seq_local} ring {ring}: k_chunk {pc.k_chunk_size} -> {_K}, q {pc.q_chunk_size}",
            flush=True,
        )
    return new


_am.MiniMaxH3Attention._sdpa_program_config = _capped
