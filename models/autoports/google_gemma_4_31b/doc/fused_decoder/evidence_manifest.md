# Fused decoder evidence manifest

Final evidence freeze: 2026-07-11 UTC. Git HEAD before the stage checkpoint is
`5fa49e9fa2589c64e74771c9759b674de19a4e42`.

## Bound source

| File | SHA-256 |
|---|---|
| `tt/fused_decoder.py` | `941dd1d16b64246111e8402d875cbb7fc1cc6bbf6b33465ff0ba97103df90af6` |
| `tt/functional_decoder.py` | `2f8a26cbdd8ebb46c8ba478080c963cd288c7936854a7023227ad58d69a144f2` |
| `tests/test_fused_decoder.py` | `29b99b8c51d8bc3f5052da35f37477267e66215627ece3e100e012071b3eee06` |
| `tests/test_functional_decoder.py` | `2fd1278ecd6703a62ba6292fa67644c1c7d3e7bc97b4d7877c1b3d9584d2d6be` |

The final standard, watcher, and exact-context distinct-token logs print these
hashes before invoking pytest. The two earlier capacity and four profiler logs
bind the identical final runtime source hash and the prior fused-test hash
`bef5374cd5ec09948fca3690d0ddbade9694cc4be748370acf41d49f79712b9d`;
the only later test-source delta is the appended inherited-test wrapper whose
own exact-context run is bound to the delivered hash above.

## Required gates

| Artifact | Result | SHA-256 |
|---|---|---|
| `standard_suite_final.log` | 23 passed, 9 gated skips | `a9f5fcc294cc46d96516d6caedf2fb324b123c78d1ce4332b485c72fbb9a151e` |
| `long_context_262144_final.log` | 2 passed | `c265d8aa3ea73f69d0f44fc5bad8f8db28e62b12ca0f327e711bb5773a0fd041` |
| `long_nonaligned_262113_final.log` | 2 passed | `cfd04e8e70d5321ea7792c523332f76874a24247b75cabc9a6715aad0bb1259a` |
| `watcher_final.log` | 4 mutable-trace cases passed | `7b7db74fb7f77d6002a48187898ce3c22b210d13cc155e4cca536659b3f0946c` |
| `watcher_final/generated/watcher/watcher.log` | clean attach/check/detach | `007556b5a38d46bb85b692cad783350cab4c92084510dce34ae62939cd0c1ff0` |
| `exact_context_distinct_262144_final.log` | 2 distinct-token HF gates passed | `f9b28879af342a1dc8e6394675d307639d464e696cd2483fc0c1c10d72c94e19` |

## Canonical profiler evidence

Every `profile_command_final.log` records the exact node, UTC time, Git HEAD,
and four source/test hashes. Tracy was written directly to the delivered
`tracy/<kind>/<mode>/` directory. The newest raw Tracy ops CSV was copied to
the stable `*_ops.csv` name in that same directory, and `tt-perf-report` was
run from that stable raw file directly to the adjacent filtered CSV/text files.

| Path | Device time / ops | Raw SHA-256 | Filtered CSV SHA-256 | Text SHA-256 |
|---|---:|---|---|---|
| `tracy/sliding/prefill` | 3426.812 us / 26 | `71d1d1583d8a082ccaa8e44f9788fdd3aab480d1ae8ca66549b7ec5b97226e78` | `dc81b69bed40ffc2c064866243043e89b3f47e7d96511148618c13df74daef3c` | `c5502645f2153a29422b2d6327ba2ba2092e2a20cf325273666a60cda8ec7db3` |
| `tracy/full/prefill` | 4192.270 us / 23 | `a3c04b4a9c976d71d842e700dce38259672d68640a7c312fe85f901c5cea337b` | `0d10a8adb8d9d420ae16f32f9ee669a0c5f8ade7f7b48b121ca192f226584d91` | `23d3ce39fef93f4dd9452b249d0e374fe94896a20d9743c7e1597524bd13cefe` |
| `tracy/sliding/decode` | 2560.197 us / 40 | `644ad58c95db959681cf1721d50282243975e9e6238452cc09e9f7f825e102d3` | `84b3a77b7c65a9ee1085a57d3ac4e882b22b1a83eebd210bab0e6f00ee503643` | `bac3a66ac48d91d1f91ea0b79796de7395488a42954a6773e4177b5518840797` |
| `tracy/full/decode` | 2880.903 us / 39 | `f8b5b971861881b35749d2493326bb8becc4a74e692e9086d059c09a3cb47fee` | `ef31281ef702515ea3993837ccf6bf2d3f1fc9814fb4117deba7eaa7e5c744de` | `babf9c09b5ba7bd3a22559eeba65a25ba9bbfcdbf05a5c13dd21caccb77b0af1` |

## AutoFix and candidate evidence

| Artifact | Purpose | SHA-256 |
|---|---|---|
| `AUTODEBUG.md` | ranked review diagnosis | `dd80283b38e07b7c56bb2f24e1cc28be636a29e561b0a6a4083d032172232a0b` |
| `AUTOFIX.md` | verified/refuted hypothesis ledger | `d0296e33e438c12b4274c8880148d94da5191139f619ca62bd6f4c0ff58e5923` |
| `stage_review_final.md` | independent final `clean-pass` | `9c621254a95ab5be9346dc02a3fe0281102f8873f4e6f866abbffc289bb54183` |
| `stage_review_current.md` | live-worktree independent `clean-pass` refresh | `6a6402e26d12d97f36038fe59dd465828b1aedc688fee133f1d0a4f7fbea0732` |
| `candidates/post_projection_slice_repeated/summary.md` | 12-sample paired decode A/B | `d8888d794f2e91de02a3b806a525777e995a57ecd2cf7912b07ceeac042d6e95` |
| `candidates/post_projection_slice_repeated/repeated_ab.log` | ordinary synchronized A/B | `cf281f276a7c7da7b4d2251591efa873ef5cf93babe845128dd412c1486dc274` |

Long-GELU real-weight gate logs and hashes:

| Candidate | SHA-256 |
|---|---|
| F4 | `4cf05de9fcb4591a4db1691c9a0ec932a033d7d76ca0b2c57417dd0986040575` |
| F2 | `3648eaf6d38b9689077a3391c6339f93b03e93598dd9f328bb98c6f607a194b1` |
| F1 | `f07c53acf77c3efd27e3b514c7a1f457c2ce6266beba8296348eda2be3a7e94f` |
| C2048 | `10964dcf9097e1c233cd22e279b923548d7f7046edfc874229f79c06cae044ef` |
| C1024 | `1a103f71122187a0225b1106bbbaec5e3994bb5dbdde542d02949e9b980373b0` |
| C128 | `fa379382972fe7773628638502dca3dec3ea92d3aff8bd6e042027811068b70b` |

Candidate `tt-perf-report` files were regenerated directly beside their own raw
CSV inputs. No candidate console output points at a canonical selected path.
