"""Root-cause repro #2: is the TT AuT encoder (encode_mel) consistent + finite across
different mel lengths and across interleaved calls? The decoder was proven stable, so a
bad audio embedding (NaN / wrong magnitude / state-dependent) is the remaining suspect for
the server's empty outputs on certain lengths."""
import os, sys
import numpy as np, torch, ttnn
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tt"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "reference"))
import audio_encoder as tt_enc          # noqa
import audio_encoder_ref as ref         # noqa


def stats(name, e):
    a = e.float()
    print(f"  {name:14s} shape={tuple(a.shape)} finite={bool(torch.isfinite(a).all())} "
          f"mean={a.mean():.4f} std={a.std():.4f} absmax={a.abs().max():.3f}")
    return a


def main():
    snap = "/root/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots"
    snap = os.path.join(snap, os.listdir(snap)[0])
    w = ref.load_audio_tower_weights(snap_dir=snap, dtype=torch.float32)
    dev = ttnn.open_device(device_id=0, trace_region_size=200000000, l1_small_size=65536)
    try:
        p = tt_enc.preprocess_weights(w, dev)
        mel = torch.from_numpy(np.load("/golden/input_features.npy")).float()  # (128,1200)=12s
        A = mel                                   # 12s  (12 chunks)
        Bs = mel[:, :600].contiguous()            # 6s   (6 chunks)
        Bl = mel.repeat(1, 3)[:, :3000].contiguous()  # 30s (30 chunks) -- the length the server empties on
        # reference (CPU) encoder output for A, as a correctness anchor
        ref_A = ref.encode(A, w)
        print("lengths: A=%d Bs=%d Bl=%d mel-frames" % (A.shape[1], Bs.shape[1], Bl.shape[1]))
        eA = stats("A (12s)", tt_enc.encode_mel(A, p, dev))
        stats("Bs (6s)", tt_enc.encode_mel(Bs, p, dev))
        eBl = stats("Bl (30s)", tt_enc.encode_mel(Bl, p, dev))   # <-- check 30s finite/sane
        eA2 = stats("A again", tt_enc.encode_mel(A, p, dev))
        # interleave then re-check A
        _ = tt_enc.encode_mel(Bl, p, dev); _ = tt_enc.encode_mel(Bs, p, dev)
        eA3 = stats("A after Bl,Bs", tt_enc.encode_mel(A, p, dev))

        def pcc(x, y):
            x, y = x.flatten().double(), y.flatten().double()
            return float(torch.corrcoef(torch.stack([x, y]))[0, 1])
        print(f"PCC A vs CPU-ref      = {pcc(eA, ref_A):.4f}")
        print(f"PCC A vs A-again      = {pcc(eA, eA2):.4f}  (1.0 => deterministic)")
        print(f"PCC A vs A-after-mix  = {pcc(eA, eA3):.4f}  (<1 => encoder STATE LEAK)")
        print(f"Bl(30s) finite        = {bool(torch.isfinite(eBl).all())}  absmax={eBl.abs().max():.3f}")
        print("RESULT:", "ENCODER OK" if (pcc(eA, eA3) > 0.999 and torch.isfinite(eBl).all()) else "ENCODER PROBLEM")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
