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
