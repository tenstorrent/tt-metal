"""Audit conv3d blocking tables for the vol2col_rm chunk-straddle hazard (#149).

Hazard: T_out_block*H_out_block*W_out_block > 64 and not a multiple of 32.
"latent" = the table blocking is safe, but a smaller T_out_block (the T-relaxed path clamps it
to T_out) would hit the hazard.
Run: python tt-project/t149/audit_blockings.py (CPU only).
"""

from models.tt_dit.models.audio_vae.minimax_h3.blockings_minimax_h3_audio import register_h3_audio_blockings
from models.tt_dit.models.vae.minimax_h3.conv_minimax_h3 import _H3_ENCODER_BLOCKINGS
from models.tt_dit.utils.conv3d import _BLOCKINGS, _DEFAULT_BLOCKINGS, _FP32_BLOCKINGS


def hazard(t, h, w):
    n = t * h * w
    return n > 64 and n % 32 != 0


def audit(name, table):
    hits, latent = [], []
    for k, v in table.items():
        _, _, t, h, w = v
        if hazard(t, h, w):
            hits.append((k, v, t * h * w))
        elif any(hazard(tt, h, w) for tt in range(1, t)):
            latent.append((k, v, [tt for tt in range(1, t) if hazard(tt, h, w)]))
    print(f"{name}: {len(table)} entries, {len(hits)} hits, {len(latent)} latent")
    for k, v, n in hits:
        print(f"  HIT    {k} -> {v} ({n} patches)")
    for k, v, ts in latent:
        print(f"  latent {k} -> {v}: T_out_block in {ts} would hit")
    return hits, latent


if __name__ == "__main__":
    register_h3_audio_blockings()
    audit("_BLOCKINGS", _BLOCKINGS)
    audit("_BLOCKINGS mesh 4x8", {k: v for k, v in _BLOCKINGS.items() if k[:2] == (4, 8)})
    audit("_DEFAULT_BLOCKINGS", _DEFAULT_BLOCKINGS)
    audit("_FP32_BLOCKINGS (+H3 audio)", _FP32_BLOCKINGS)
    audit("_H3_ENCODER_BLOCKINGS", _H3_ENCODER_BLOCKINGS)
