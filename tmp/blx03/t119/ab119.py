"""Run test_vae_ltx_trace_ab.py with one conv3d blocking patched into _BLOCKINGS and log every LTX conv key.

T119_KEY names the patched key (KEYS below, or "none"), T119_BLK its blocking "Cin,Cout,T,H,W".
Each LTX conv's (mesh, Cin, Cout, kernel, T, H, W) and chosen blocking prints once as T119_CONV; T119_HIT
says whether the patched key was looked up. Usage: python ab119.py <pytest args>
"""
import os
import sys

import pytest

from models.tt_dit.models.vae import vae_ltx
from models.tt_dit.tests.models.wan2_2.bruteforce_conv3d_sweep import prefetch_shard_fits
from models.tt_dit.utils import conv3d

KEYS = {
    "s0ups": (4, 8, 128, 1024, (3, 3, 3), 21, 5, 4),
    "s4res2x4": (2, 4, 128, 128, (3, 3, 3), 147, 136, 120),
}
name = os.environ.get("T119_KEY", "none")
key = KEYS.get(name)
if key is not None:
    blk = tuple(int(v) for v in os.environ["T119_BLK"].split(","))
    assert prefetch_shard_fits(*blk, key[4], key[2]), f"{blk} gets no L1 prefetch shard"
    conv3d._BLOCKINGS[key] = blk
    conv3d._BLOCKINGS_BY_SPATIAL = None  # rebuilt on next lookup; the T-relaxed index must see the patch
print(f"T119_ARM key={name} {key} blk={conv3d._BLOCKINGS.get(key)}", flush=True)

seen = {}
_orig = vae_ltx.get_conv3d_config


def _logged(cin, cout, kernel, dtype, grid_size, *, h_factor=1, w_factor=1, T=0, H=0, W=0):
    cfg = _orig(cin, cout, kernel, dtype, grid_size, h_factor=h_factor, w_factor=w_factor, T=T, H=H, W=W)
    hf, wf = conv3d._blocking_mesh_override(h_factor, w_factor, cin, cout, kernel, T, H, W)
    k = (hf, wf, cin, cout, tuple(kernel), T, H, W)
    if k not in seen:
        seen[k] = (cfg.C_in_block, cfg.C_out_block, cfg.T_out_block, cfg.H_out_block, cfg.W_out_block)
        print(f"T119_CONV {k} -> {seen[k]}", flush=True)
    return cfg


vae_ltx.get_conv3d_config = _logged
rc = pytest.main(sys.argv[1:])
print(f"T119_HIT key={name} hit={key in seen if key else 'n/a'} blk_used={seen.get(key)}", flush=True)
sys.exit(rc)
