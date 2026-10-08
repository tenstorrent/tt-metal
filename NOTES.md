# t298: finish ttp/ltx23-main-pr (user #194)

Branch ttp/ltx23-main-pr (base origin/main 80b1cd689d0) pushed at df9e5ecaac6 (verified by git ls-remote).

## Commits added in t298 (source -> new)
- eee3baf7c0d (parallel AAC) -> 5565232f12c: AAC encode runs on a 1-worker thread beside the video encode, on both export paths. Service/export-only.
- 4e7d0cefcdb (audio part) -> 05401286709: main already traced the vocoder and the BWE vocoder by default; only the mel-VAE ran eager. It now traces by default on a traced pipeline (LTX_VAE_TRACE=0 / LTX_VOC_TRACE=0 force eager). Moves the e2e number (audio decode is inside the timed section).
- e7588fb8718 (video.py x264 only) -> df9e5ecaac6: ultrafast crf 20 (was veryfast crf 23). Own commit, droppable (user #200). Files ~3.5x larger, Y PSNR vs source 45.6 -> 47.9 dB; encode ~0.15 s vs 0.65 s on 1080p 145 frames. Service/export-only.

## Gains split (user #202)
- Changes the number the standard e2e test reports: QK-RoPE fusion 70157c213e6, host-thread seeded noise bfc93527e92, S2 prompt reuse e6b23ad361c, V2A pad-mask skip dca60c1c777, mel-VAE trace 05401286709.
- Service/export-only (the test does not time mp4 export): ultrafast x264 df9e5ecaac6, parallel AAC 5565232f12c.

## Left out
sync removal 4194cd98852, resident upsampler ab67c9e86b2, async mp4 encode 63902277007 (earlier decision); from e7588fb8718: yuv zero-copy from_numpy_buffer, LTX_EXPORT_* kill switches, Gemma trace gate; from 4e7d0cefcdb: prep_run/clone_prep_inputs=False, vocoder _tpad_mask_cache.clear(), LTX_BWE_TRACE, sources=[...] cache args (all tied to ltx-rt's capture-only warmup).

## Checks
- New CPU tests: test_ltx_export_latency.py (7) + test_ltx_audio_trace_gates.py (3): 10 passed.
- test_rope_active_core_source.py (--noconftest, CPU): 2 passed.
- test_rope_active_core_cache.py: device-only (mesh_device fixture, opt-in C16_CACHE_CHECK=1) -> not run (no device work in this task).
- C++ build check of 70157c213e6 ops: NOT DONE. No new build dir allowed (project footprint ~102 GB > 100 GB cap). Tried a syntax-only compile (tt-project-notes/t298/syntax_check.py) with the root build_Release flags against the worktree; it fails on missing third-party headers because the root's .cpmcache is gone (fmt only in build_Release/include; nlohmann_json and nanobind not found). The device kernel dit_rmsnorm_fused_compute.cpp is JIT-only anyway.

## e2e command (BH 4x8 ring, 8+3 default, unmodified)
pytest "models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py::test_pipeline_distilled[blackhole-4x8sp1tp0nl2_ring_is_fsdp0-True]"
Same id on main (80b1cd689d0) and on ttp/ltx23-main-pr (df9e5ecaac6).
