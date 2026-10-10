# t286 pytest plugin (-p t286_skipvaewarm): construction-time VAE warmup decodes only the canvases
# this test can reach, not all 93 a server must serve (~3.5 min per process on 4x8), so each broker
# job fits the 600 s cap. The measured call is unchanged: the test's untimed priming call at the same
# shape runs first. The full-wave canvas (more latents than one chunk) is still decoded, so every
# gather-stitch slot is compiled as before.
import os


def pytest_configure(config):
    from PIL import Image

    from models.tt_dit.pipelines.minimax_h3 import pipeline_minimax_h3 as pm
    from models.tt_dit.pipelines.minimax_h3.packing import resolve_canvas_size

    keep = {(768, 1344), resolve_canvas_size(1344, 768)}
    kf = os.environ.get("MINIMAX_H3_TURBO_KEYFRAME")
    if kf:
        keep.add(resolve_canvas_size(*Image.open(kf).size))
    orig_warm = pm.MiniMaxH3Pipeline._warm_vae_decode

    def warm(self):
        vae = self._vae
        ratio = self.vae_config.spatial_compression_ratio
        chunk = vae.decode_unit_shape()[0]
        orig_decode = vae.decode
        skipped = []

        def decode(z, **kw):
            canvas = (z.shape[-2] * ratio, z.shape[-1] * ratio)
            if canvas in keep or z.shape[2] != chunk:
                return orig_decode(z, **kw)
            skipped.append(canvas)

        vae.decode = decode
        try:
            orig_warm(self)
        finally:
            del vae.decode
        print(f"[t286] VAE warmup: kept {sorted(keep)} + full-wave canvas, skipped {len(skipped)} canvases", flush=True)

    pm.MiniMaxH3Pipeline._warm_vae_decode = warm

    # T286_WARM_DEADLINE_S: the construction warmup (93 VAE canvases, 12 audio lengths, 28 prompt
    # layouts, 26 denoise rungs) needs several cold jobs to compile. Stop it between warm items once
    # the deadline passes, or fail the test if it ends too late for the generation to still fit, so
    # the job fails cleanly with the mesh closed instead of hitting the broker's 600 s kill. Compiled
    # kernels persist in TT_METAL_CACHE, so the next job gets further.
    deadline = float(os.environ.get("T286_WARM_DEADLINE_S", "0"))
    if deadline <= 0:
        return
    import time
    import types

    import tqdm as tqdm_mod

    start = time.monotonic()
    late = float(os.environ.get("T286_WARM_LATE_S", deadline))

    def bounded_tqdm(iterable=None, *args, **kwargs):
        for item in tqdm_mod.tqdm(iterable, *args, **kwargs):
            if time.monotonic() - start > deadline:
                raise RuntimeError(f"[t286] warm deadline {deadline:.0f} s passed in {kwargs.get('desc')}")
            yield item

    pm.tqdm = types.SimpleNamespace(tqdm=bounded_tqdm)
    orig_warmup = pm.MiniMaxH3Pipeline.warmup

    def warmup(self, *args, **kwargs):
        orig_warmup(self, *args, **kwargs)
        took = time.monotonic() - start
        print(f"[t286] construction warmup done at {took:.0f} s", flush=True)
        if took > late:
            raise RuntimeError(f"[t286] warmup ended at {took:.0f} s, past {late:.0f} s: generation would not fit")

    pm.MiniMaxH3Pipeline.warmup = warmup
