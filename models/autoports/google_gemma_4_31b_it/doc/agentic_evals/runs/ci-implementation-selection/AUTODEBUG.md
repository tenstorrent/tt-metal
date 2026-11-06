# Missing explicit implementation dispatch input

Terminal retry42175367781 is a no-score wrong-adapter launch, not an autoport model failure. Full job log bringup/artifacts/stage12-terminal-retry-job.log line66 records impl-of-model:default. Lines1123-1125 render an empty conditional and do not append --impl. The stock catalog selects its default tt-transformers leaf, and line2007 emits a class override naming the removed stock adapter. It fails before tasks with the reported missing class. No old implementation source was inspected.

Stock route forwarding is intact: setup/dispatch-workflow.yml:101 forwards inputs.impl-of-model; shield-on-dispatch.yml:346 stores nondefault input in matrix.impl; shield-run-tests.yml:343-345 appends --impl only if nonempty. The new autoport deliberately has default_impl:false, so explicit selection is required.

Host replay of exact TTIa087be086b6e2cd6edf215dddb4c47c48463f13a dev catalog reproduces default id_tt-transformers_gemma-4-31B-it_p300x2, versus explicit id_gemma4-31b-autoport-v3_gemma-4-31B-it_p300x2 with both required autoport class overrides. Resolver source unchanged between a087 and currentHEAD. See host-check.json.

Other runs: coordinator inspected SWE live log lines66,1222,1348 and confirmed explicit implementation and actual autoport; do not cancel it. GPQA nightly live logs unavailable (404/in-progress); no conclusion from missing logs, and no blind cancellation. Its actual dispatch selection must be checked when evidence is available.
