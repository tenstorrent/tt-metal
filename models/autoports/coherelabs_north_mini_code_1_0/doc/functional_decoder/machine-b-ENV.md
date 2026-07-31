# skillexp — MACHINE B environment record

Machine role: **B** (`$shard-advise` OFF). Arms: `nofuse-noadvise` (phase 2), `fuse-noadvise` (phase 3).
Owns functional decoders for: `coherelabs_north_mini_code_1_0`, `google_gemma_4_26b_a4b_it`.
Machine B needs no tt-mlir checkout (EXPERIMENT.md: "This arm needs no tt-mlir checkout at all.")

Recorded: 2026-07-28T16:35:47Z

## Host
```
hostname: qb2-120-p04t04
nproc:    16
               total        used        free      shared  buff/cache   available
Mem:             249          24         207           0          19         224
```

## Pinned commits (resolved on this machine, verified against the experiment record)

| branch | resolved here | experiment record | match |
|---|---|---|---|
| `mvasiljevic/qb2/skillexp/base` | `b9e6c242a34` | `b9e6c242a34` | OK |
| `mvasiljevic/qb2/skillexp/fuse-advise` | `e8ae927d77e` | `e8ae927d77e` | OK |
| `mvasiljevic/qb2/skillexp/fuse-noadvise` | `03a8221501d` | `03a8221501d` | OK |
| `mvasiljevic/qb2/skillexp/nofuse-advise` | `d430985b1a9` | `d430985b1a9` | OK |
| `mvasiljevic/qb2/skillexp/nofuse-noadvise` | `51b17c3da34` | `51b17c3da34` | OK |

Verified 0 non-`.agents/` diffs between every arm and `aab03552379` — one build serves all arms.

## tt-metal build
```
HEAD:       b9e6c242a34011e3daeebab9207fbb5b79750f39  (== pinned base b9e6c242a34)
branch:     skillexp-base  (local branch tracking refs/skillexp/base)
submodule:  29125b7ad8b5513eeaa4417ed92892bf39c8bd74 models/demos/t3000/llama2_70b/reference/llama (heads/main)
submodule:  b3437e6b8bd383af24511db6d82a96e5933b6f64 tt_metal/third_party/tracy (v0.13.3-tt.0)
submodule:  44455fa126dc52460bad107e491103d7cd4caf81 tt_metal/third_party/umd (v0.9.5-101-g44455fa1)
```

## Devices
```
                      All available boards on host (UMD):
┏━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ UMD Chip   ┃            ┃            ┃            ┃ Device      ┃ Board      ┃
┃ ID         ┃ PCI BDF    ┃ PCI Dev ID ┃ Board Type ┃ Series      ┃ Number     ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ 0          │ 0000:01:0… │ /dev/tens… │ Blackhole  │ p300c       │ 000004613… │
│ 1          │ 0000:02:0… │ /dev/tens… │ Blackhole  │ p300c       │ 000004613… │
│ 2          │ 0000:03:0… │ /dev/tens… │ Blackhole  │ p300c       │ 000004613… │
│ 3          │ 0000:04:0… │ /dev/tens… │ Blackhole  │ p300c       │ 000004613… │
└────────────┴────────────┴────────────┴────────────┴─────────────┴────────────┘
                        Boards that can be reset (UMD):
┏━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
```

## Codex runner
```
codex:  codex-cli 0.145.0
auth:   Logged in using ChatGPT
path:   /home/mvasiljevic/.local/bin/codex -> ~/.nvm/versions/node/v24.18.0/bin/codex
```

## Build
```
INFO: Enable ccache: OFF
INFO: Build type: Release
INFO: Build directory: build_Release
ninja steps: [638/639]
errors:      0
=== build exit=0 2026-07-28T16:39:50Z ===
recipe:      ./build_metal.sh (run inside mvasiljevic-ttxla as uid 6002)
artifacts:   2026-07-28T16:39:49Z build_Release/lib/_ttnn.so
```

Full log: `~/skillexp-logs/build.log`

## Runner
```
python:        3.12.13 (uv-created python_env; NOTE no pip - use 'uv pip install')
openai-codex:  0.144.4 (python pkg, from .agents/requirements.txt)
codex CLI:     0.145.0 (npm @openai/codex)  [minor skew vs python pkg, benign]
import ttnn:   OK (exit 0; benign nanobind shutdown warning)
```
