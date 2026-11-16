# Cap-1024 fixture review

## Scope and disposition

Host-only review of the three prepared 512-to-1024 diagnostic fixtures: served Pytest7432 seed42, and full Requests5414 call11 seeds43 and44. **No admission or launch was performed.** No TT import, device work, or stage signoff is included.

**Disposition:** the three fixtures preserve their source inputs and trials, and both stored forecasts recompute exactly. No concrete fixture or cost error found.

## Fixture and source checks

For all three fixtures, compared the entire candidate `prompt_token_ids` and `continuations` against the referenced original 512-cap case. Both are exactly equal. Candidate metadata sampling is identical to source sampling, the candidate seed matches the selected row in the source run, and metadata records cap512→1024. Each source sample contains512 generated token IDs with511 timed decode steps. The candidate manifest uses `branches: []`, `free_run_tokens: 1024`, and `free_run_trials: 1`, so it requests one free-running extension and no teacher-forced branch.

The exact selected source observations are: served Pytest7432 call19 seed42 (source verdict progressing/inconclusive at cap); full Requests5414 call11 seeds43 and44 (both source verdicts progressing/inconclusive at cap). The fixture metadata’s source-case and source-result SHA-256 values match the files on disk.

| Mode and selected trial | Candidate fixture SHA-256 | Original source fixture SHA-256 | 512-cap source result SHA-256 |
|---|---|---|---|
| Served Pytest7432 call19 seed42 | `feebd5a9da150a6cf6ec67de02822a2a11338e84b8ab77a171e593d30a5adf4f` | `7ca8039708a12c24acd3c2cfb1c2c51b8946e99df118576545605efad6225bf4` | `21b2fb860e3dba265f76605c5807f7b89c32d6c76d2428444e42e95047c97639` |
| Full Requests5414 call11 seed43 | `1850ce0f5b3b322ac4b147aacefc1c1a6fe342a9d7440354dda7c659a96c07c2` | `705c84003084ef0866174aa7d7840b54ca8287a4daa5f3a0b98c5a7dbc88aa74` | `ef5a245dd5579605de4ff1ece121865078b20c869e9608149ce49742f0512ac3` |
| Full Requests5414 call11 seed44 | `2327d31bd7c4aef5ff2156ddbafc71dbaa39b4dcd9e74c1705e4937eb154b929` | `705c84003084ef0866174aa7d7840b54ca8287a4daa5f3a0b98c5a7dbc88aa74` | `ef5a245dd5579605de4ff1ece121865078b20c869e9608149ce49742f0512ac3` |

## Forecast recomputation

Recomputed each mode as one same-mode measured load plus each selected sample’s observed prompt/prefill wall (`diagnostic_wall_s_including_prefill - decode_wall_s`) and a 1,023-step decode replay extrapolated from its 511 measured decode steps:

`load_s + Σ[prefill_s + decode_wall_s × 1023 / 511]`.

| Mode | Measured load | Scaled sample work | Recomputed total | Stored prediction | Stored range |
|---|---:|---:|---:|---:|---:|
| Served, one trial | 145.357220 s | 28.890929 s | 174.248149 s / 0.0484023 h | 0.0484023 h | 0.0459947–0.0524149 h |
| Full, two trials | 78.623041 s | 118.654696 s | 197.277737 s / 0.0547994 h | 0.0547994 h | 0.0449115–0.0712792 h |

The forecast ranges apply 0.7× to 1.5× uncertainty to the extrapolated sample-work component while retaining the one measured load. These are diagnostic time estimates, not serving-performance claims.

## Reviewed artifact hashes

- `served-candidate-cases.json`: `33d93f355ff768860f4136e6adbd66749b594ba8793ccd7ced1c76c7422dd2e1`
- `full-candidate-cases.json`: `a2099458e1dfe4e50597cadc1c5a7664916af48280f5421654452966dc614b34`
- `served-prediction.json`: `af3059c7bf098181469d641082fb1c140aa101244c46982e4c63563d08ee6542`
- `full-prediction.json`: `75c3b558f7684eb044cd4f4886a9ba6c74cbe7cc9ea14e4069d4eb59cadd89fc`
- Served load record: `54200eb563a980f67d2f49428ff98ccf2419effa0967045fd9425749b53d7402`
- Full load record: `7ccfc36c3db10085a06d789c1b6a0d29198c1a06ef2136031937a98b34ca3244`
- Source six-caret trial triage: `afca6672f81b58344251c44e2ff06a132bcc66ba6d6fc672de812ff26f6872fe`
- Machine-readable review: `review-d.json`, SHA-256 `a5b52b8e29543b68de18e1ff6042808eb2916110ab000743d65d6dc28db8e491`
