# Stage 09 (vLLM integration) runner artifacts

This is the `readiness_vllm/` directory exactly as the **vLLM-integration**
stage left it, at commit `ab0fbebb4b1`. It is the evidence behind
`doc/vllm_integration/README.md`'s numbers: primary single-user 128/128/1 TTFT
273.83 ms and TPOT 45.001 ms (22.22 t/s/u), CI serving-burst 100/100/32 output
throughput 137.08 tok/s.

It was copied here by the **optimized-vLLM** stage (stage 10) before that stage's
own `run_vllm_server` runs overwrote the live `readiness_vllm/` directory, so
that no earlier stage's committed evidence was destroyed in place. The live
`models/autoports/zai_org_glm_4_7_flash/readiness_vllm/` now holds stage 10's
after-arm serving evidence; see `doc/optimized_vllm/README.md`.

`server.log` is not here for the same reason stage 09 did not commit it: it
exceeds the repo's 500 KB file limit and is a debug log, not stage evidence.
