"""Audit CPU/host fallbacks in the forward path. Sets throw_exception_on_fallback
and runs each block; any op that falls back to host raises and is reported.
Run: TT_DEVICE=1 /home/yito/work/run_zaya.sh python models/demos/zaya1_8b/tests/run_fallback_audit.py
"""
import os
import traceback
import torch
import ttnn

from models.demos.zaya1_8b.tt.model_args import ZayaWeights
from models.demos.zaya1_8b.tt import standard as S
from models.demos.zaya1_8b.tt.moe import ZayaMoEBlock
from models.demos.zaya1_8b.tt import cca as CCA

GOLDEN = os.path.join(os.path.dirname(__file__), "..", "reference", "golden")


def g(n):
    return torch.load(os.path.join(GOLDEN, f"{n}.pt"), weights_only=False)


def run(name, fn):
    try:
        fn()
        print(f"  [device-ok]  {name}")
    except Exception as e:
        msg = str(e).splitlines()[0][:160]
        print(f"  [FALLBACK?]  {name}: {type(e).__name__}: {msg}")


def main():
    ttnn.CONFIG.enable_fast_runtime_mode = False   # fast mode bypasses fallback checks
    ttnn.CONFIG.throw_exception_on_fallback = True
    ttnn.CONFIG.enable_logging = True
    print(f"throw_on_fallback={ttnn.CONFIG.throw_exception_on_fallback} fast_runtime={ttnn.CONFIG.enable_fast_runtime_mode}")

    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape((1, 1)))
    try:
        w = ZayaWeights()
        ids = g("inputs")["input_ids"]
        hid_t = g("L1_zaya_block")["in"][0].float()       # [1,6,2048]
        S_len = hid_t.shape[1]
        hidden = S.to_dev(hid_t, device)

        emb = S.Embedding(device, w.embed())
        norm = S.RMSNorm(device, w.att(0, "input_norm.weight"))
        head = S.LMHead(device, w.embed())
        moe = ZayaMoEBlock(device, w, 1)
        moe3 = ZayaMoEBlock(device, w, 3)                 # EDA-enabled
        attn = CCA.CCAAttention(device, w, 0, S_len)

        run("embedding", lambda: emb(ids))
        run("rmsnorm", lambda: norm(hidden))
        run("lm_head", lambda: head(hidden))
        run("moe_block(L1)", lambda: moe.forward(hidden))
        run("moe_block(L3,EDA)", lambda: moe3.forward(hidden, moe.forward(hidden)[1]))
        run("cca_qkv", lambda: attn.qkv.forward(hidden))
        run("cca_attention", lambda: attn.forward(hidden))
        # individual suspect ops
        x = S.to_dev(torch.randn(1, S_len, 17), device)
        run("max(dim=-1,kd)", lambda: ttnn.max(x, dim=-1, keepdim=True))
        run("eq(broadcast)", lambda: ttnn.eq(x, ttnn.max(x, dim=-1, keepdim=True)))
        run("softmax(17)", lambda: ttnn.softmax(x, dim=-1))
        y = S.to_dev(torch.randn(1, 2, S_len, 128), device)
        run("repeat_interleave", lambda: ttnn.repeat_interleave(y, 4, dim=1))
        run("pad(rowmajor seq)", lambda: ttnn.pad(ttnn.to_layout(hidden, ttnn.ROW_MAJOR_LAYOUT), [(0, 0), (1, 0), (0, 0)], value=0.0))
        run("argmax(vocab)", lambda: ttnn.argmax(S.to_dev(torch.randn(1, 1, 262272), device), dim=-1))
        run("to_layout(tile->rm->tile)", lambda: ttnn.to_layout(ttnn.to_layout(hidden, ttnn.ROW_MAJOR_LAYOUT), ttnn.TILE_LAYOUT))
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
