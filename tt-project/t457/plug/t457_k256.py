"""t457 noise-floor control: dense exp-ring SDPA with k_chunk capped at 256 (default 512), same precision.

Only the k-chunk accumulation order changes (more running max/sum rescales, different bf16 rounding). PSNR of this
dense arm vs the stock dense arm is the trajectory floor for any equal-precision attention substitution.
"""
import os

from models.tt_dit.models.transformers.minimax_h3 import attention_minimax_h3 as _am

_init = _am.MiniMaxH3Attention.__init__
_K = int(os.environ.get("T457_KMAX", "256"))


def _patched(self, *a, **kw):
    _init(self, *a, **kw)
    self.exp_ring_max_k_chunk = _K


_am.MiniMaxH3Attention.__init__ = _patched
_seen = {}
_build = _am.MiniMaxH3Attention._build_exp_sdpa_program_config


def _logged(self, seq_local):
    pc = _build(self, seq_local)
    if seq_local not in _seen:
        _seen[seq_local] = 1
        print(f"[t457] exp sdpa seq_local {seq_local}: {pc}", flush=True)
    return pc


_am.MiniMaxH3Attention._build_exp_sdpa_program_config = _logged
