# Device time of ttnn.embedding_bw (reshuffle_rows), under `python -m tracy -r -p --no-web-server -o <dir> prof_rr.py
# <cases>`; a signpost precedes the R3_NMEAS measured calls of every case, after 2 warm-up calls.
import gc, os, sys
import torch
import ttnn
from tracy import signpost

N_WARM, N_MEAS = 2, int(os.environ.get("R3_NMEAS", "6"))


def case(b, s, e, v, odt=ttnn.bfloat16):
    def setup(d):
        g = torch.Generator().manual_seed(1234)
        ti = ttnn.from_torch(torch.randint(0, v, (b, s), generator=g), dtype=ttnn.uint32, device=d)
        tw = ttnn.from_torch(torch.randn((v, e), generator=g), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=d)
        tg = ttnn.from_torch(torch.randn((1, 1, b * s, e), generator=g), dtype=odt, layout=ttnn.TILE_LAYOUT, device=d)
        return (ti, tw, tg)

    def run(d, ti, tw, tg):
        return ttnn.embedding_bw(ti, tw, tg, dtype=odt)

    return setup, run


CASES = {"emb_2x1024x4096x3200": case(2, 1024, 4096, 3200), "emb_1x2048x2048x32000": case(1, 2048, 2048, 32000),
         "emb_8x128x768x30522": case(8, 128, 768, 30522), "emb_2x64x160x96": case(2, 64, 160, 96)}


def main():
    names = sys.argv[1:] or list(CASES)
    dev = ttnn.open_device(device_id=0)
    try:
        for name in names:
            setup, run = CASES[name]
            gc.collect()
            try:
                args = setup(dev)
                for _ in range(N_WARM):
                    run(dev, *args)
                ttnn.synchronize_device(dev)
                signpost(header=name)
                for _ in range(N_MEAS):
                    run(dev, *args)
                ttnn.synchronize_device(dev)
                ttnn.ReadDeviceProfiler(dev)
                print(f"CASE {name} ok", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"CASE {name} FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
