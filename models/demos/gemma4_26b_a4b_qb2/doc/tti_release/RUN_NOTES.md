# Gemma 4 26B A4B TTI release evidence

Final bounded release: [tt-shield 38381737849](https://github.com/tenstorrent/tt-shield/actions/runs/38381737849), job [115243700311](https://github.com/tenstorrent/tt-shield/actions/runs/38381737849/job/115243700311), 13–14 October 2026.

## Provenance

- Requested tt-metal head: `2ce68039f08fed5f8ea7be77e1a7c6b03c8a35e9`.
- Image tt-metal commit: `78e87bd9380740a522ec4b5ab62a46c9498d7b35`.
- The commits have no difference under `tt/`, `config/`, or
  `vllm_metadata.json` in this model directory; intervening changes are tests
  and CI harness only.
- tt-inference-server: `04c29f4aa46bc3fa049a49d622107a711bdc87de`.
- vLLM TT plugin: `0a76258e14ca7237067ca9a22ec02e257bee570a`.
- Shield `main`: `78c50acf1fae34aa70f2122e85dc9d38bf51c0ce`.
- Reused image:
  `ghcr.io/tenstorrent/tt-shield/vllm-tt-metal-src-dev-ubuntu-22.04-amd64:0.23.0-78e87bd9380740a522ec4b5ab62a46c9498d7b35-0a76258-112700448945`.
- Source artifact SHA-256: report
  `4c993d53c1f01de6ac2387a15719018dcf9ec272669cabc07f9e36ecbdd40b4e`;
  report data
  `f21228159a3d3d074653a04fbc65ed97c705427bf389cbffaf06fcc9bf581f87`.
  The checked-in copies normalize the missing trailing newline only.

## Result

- Shield status: `FAIL`; model status: `FUNCTIONAL`.
- Serving: functional tier passed; S128/O128/C1 was 89.8 ms TTFT and
  53.4 output tokens/s/user.
- GPQA-Diamond: 8/10, pass.
- Terminal-Bench 2: 2/5, pass.
- SWE-bench Verified: 0/5, fail. All five trials reached the 7,200-second
  agent timeout and all post-timeout verifiers scored 0.

The model server and device stayed healthy. SWE trajectories show repeated
agent/tool loops and consumed 30.14M input plus 448.7K saved output tokens. The
earlier focused 2/5 run used a different autoport/runtime and an experimental
BFP8 policy, so it is not current-path release evidence. A current-path BFP8
probe remains an unverified follow-up, not a release fix.

The generated AI job summary called the job green despite the five evaluator
timeouts; the retained release report and raw Harbor results are authoritative.

The generated final report and JSON are retained beside this file. No unchanged
release rerun was dispatched.
