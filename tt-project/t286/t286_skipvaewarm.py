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

    # T286_RUNGS: warm and capture only these denoise rungs plus the top one, not all 26 (warm
    # rungs alone take ~8 s each to bind and again to capture, past the 600 s job cap). The ladder is
    # swapped only inside the walk, so buffer sizes and the top rung match the served config. Every
    # program the clip runs must still compile before trace capture, so a clip landing on an unwarmed
    # rung fails instead of compiling under live traces.
    rungs = {int(r) for r in os.environ.get("T286_RUNGS", "").split(",") if r}
    if rungs:
        orig_walk = pm.MiniMaxH3Pipeline._warm_denoise_buckets
        orig_select = pm.MiniMaxH3Pipeline._select_bucket
        orig_envelope = pm.served_envelope

        def walk(self, *args, **kwargs):
            ladder = self.bucket_ladder
            kept = tuple(r for r in ladder if r in rungs or r == ladder[-1])
            print(f"[t286] denoise rungs: warming {kept} of {len(ladder)}", flush=True)
            self.bucket_ladder = kept
            try:
                return orig_walk(self, *args, **kwargs)
            finally:
                self.bucket_ladder = ladder

        def select(self, seq_len):
            rung = orig_select(self, seq_len)
            if not self._warming:
                print(f"[t286] seq_len {seq_len} -> rung {rung}", flush=True)
                if rung not in rungs:
                    raise RuntimeError(f"[t286] rung {rung} (seq_len {seq_len}) was not warmed; set T286_RUNGS")
            return rung

        # Prompt-encoder layouts: the no-keyframe one and those sharing a vision-tower program key
        # (padded patches, ring or windowed) with this test's canvases, mirroring served_keyframe_layouts.
        def layout_key(n_keyframes, canvas, alignment):
            total = n_keyframes * 4 * (canvas[0] // 32) * (canvas[1] // 32)
            padded = -(-total // alignment) * alignment
            return padded, n_keyframes == 1 and padded == total

        def envelope(task, **kwargs):
            alignment = kwargs.get("patch_alignment")
            wanted = {layout_key(n, c, alignment) for n in (1, 2) for c in keep} if alignment else set()
            for n_keyframes, canvas in orig_envelope(task, **kwargs):
                if canvas is None or layout_key(n_keyframes, canvas, alignment) in wanted:
                    yield n_keyframes, canvas

        pm.MiniMaxH3Pipeline._warm_denoise_buckets = walk
        pm.MiniMaxH3Pipeline._select_bucket = select
        pm.served_envelope = envelope

        orig_call = pm.MiniMaxH3Pipeline.__call__

        def call(self, *args, **kwargs):
            before = self.mesh_device.num_program_cache_entries()
            out = orig_call(self, *args, **kwargs)
            if not self._warming:
                added = self.mesh_device.num_program_cache_entries() - before
                print(f"[t286] generation compiled +{added} programs", flush=True)
            return out

        pm.MiniMaxH3Pipeline.__call__ = call

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
