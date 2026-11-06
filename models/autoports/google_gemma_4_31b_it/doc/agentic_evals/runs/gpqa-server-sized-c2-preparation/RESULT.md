# Conservative GPQA10/C2 server-sized candidate

Prepared gpqa10-c2.patch against2c4a2628b974acc7ebc7a4fa867db68f0b108019 in isolated /home/mvasiljevic/gemma4-31b-pipeline-v3/tti-gpqa-c2-current. Enables GPQA CI_NIGHTLY10, disables Terminal/SWE, explicitly retains max_concurrent2. Same frozen IDs [152,189,46,87,52,181,31,185,50,187], no list shrink. Server remainsB8/maxseq8; caps/timeouts/sampling/native parsers/cache/HF/Harbor pins and generic Docker fix are unchanged.

Actual resolved autoport + stock common command confirms exact10 IDs and num_concurrent2 with peer limits0.32 focused model/reference tests and all full hooks passed, exit0. Log bringup/artifacts/stage12-gpqa-c2-current-hooks.log; git diff --check passed. Only model eval config and model-specific expectations differ from base.

Host-only predicted/actual device hours0, no speed credit. C8 candidate is not used: complete working-state capacity has not been established by saved histories. This candidate retains observed nightly concurrency2; coordinator decides next-rung interpretation/admission. No active-publication mutation, commit, push or dispatch.

Apply this patch rather than C8 after checking base compatibility. Dispatch stock workflow=evals with immutable model/TTI SHAs and explicit vllm-git-ref=ca7a18246e4111629ed6cb4a7d1192002749fccf. Rerun confirmation after combining later source changes.
