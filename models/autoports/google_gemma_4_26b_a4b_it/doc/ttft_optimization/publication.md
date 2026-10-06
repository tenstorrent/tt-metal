# Publication and remote qualification

Local optimization and integration readiness passed independent review. Overall
completion is **blocked on publishing the tested vLLM source commit**; no remote
benchmark was dispatched, and no terminal benchmark result is claimed.

Normal local commits and non-force pushes succeeded for:

| Repository | Exact commit | Branch |
| --- | --- | --- |
| tenstorrent/tt-metal | `079b9fb26dd7300b61a83f1f02307e8690ea0a2d` | `mvasiljevic/gemma4-ttft-opt` |
| tenstorrent/tt-inference-server | `b6c06944f6d099350f8e9dac3b3e72c58f146112` | `mvasiljevic/gemma4-ttft-monorepo-compat` |
| tenstorrent/tt-shield | `4f900886b74b82f8fd028d40f1c011c0dc322070` | `mvasiljevic/gemma4-ttft-monorepo-compat` |
| tenstorrent/tt-agentic-bringup-qb2 | `05415de64c82d91ebb3c41362266c2ba4c076c26` | `mvasiljevic/gemma4-ttft-monorepo-compat` |

The QB2 reusable workflow pins that complete reviewed Shield SHA; its exact-ref
contract test passes. TT-Metal's reviewed implementation checkpoint above
contains the measured source, sealed evidence and independent review. A later
documentation-only checkpoint records this publication result; use the exact
implementation SHA above for the eventual benchmark.
No PR, force push, history rewrite, or default-branch change was made.

## Blocking evidence

The existing nested checkout is clean at
`7f72b1c6e905f5137fe3377f2e7b42738d3f271d`, branch
`gemma4-vllm-integration`, remote `https://github.com/tenstorrent/vllm.git`.

```sh
git -C /home/mvasiljevic/tt-metal/vllm push origin HEAD:refs/heads/gemma4-vllm-integration
# remote: Permission to tenstorrent/vllm.git denied to mvasiljevicTT.
# fatal: ... HTTP 403
gh api repos/tenstorrent/vllm --jq '.permissions'
# {"admin":false,"maintain":false,"pull":true,"push":false,"triage":false}
gh api repos/tenstorrent/vllm/commits/7f72b1c6e905f5137fe3377f2e7b42738d3f271d --jq .sha
# HTTP 422: No commit found for SHA
gh api repos/mvasiljevicTT/vllm
# HTTP 404: no existing personal fork accessible
```

Granting upstream push access or having an authorized maintainer publish that
exact commit would unblock the existing reviewed configuration. Alternatively,
explicit authorization to create a personal fork would permit publishing the
same commit there; the narrow repository allowlists and their tests would then
need a separately reviewed update. A new external repository was not created
by inference. An unavailable SHA is not replaced by a different implementation.

## Intended dispatch, not executed

Workflow ID `342177897`, **Model dispatch (tt-shield)**, is active in
`tenstorrent/tt-agentic-bringup-qb2`. After verifying every required ref is
available remotely:

```sh
gh workflow run 342177897 -R tenstorrent/tt-agentic-bringup-qb2 \
  --ref mvasiljevic/gemma4-ttft-monorepo-compat \
  -f model=gemma-4-26B-A4B-it -f impl-of-model=gemma4-autoport \
  -f runner-label=bh-qb-ge -f device-type=p300x2 -f workflow=benchmarks \
  -f tt-metal-git-ref=079b9fb26dd7300b61a83f1f02307e8690ea0a2d \
  -f inference-server-git-ref=b6c06944f6d099350f8e9dac3b3e72c58f146112 \
  -f vllm-repository=tenstorrent/vllm \
  -f vllm-git-ref=7f72b1c6e905f5137fe3377f2e7b42738d3f271d \
  -f docker-image= -f throttle-perf=0
```

The blank image input is deliberate: build the exact optimized sources and
validate the installed engine/plugin provenance. Record the resulting run URL,
resolved inputs, terminal conclusion and artifact/result links under the
sibling `readiness_vllm/ttft_optimization_remote/` evidence area. Monitor until
terminal and diagnose any in-scope integration failures. Local cleanup is
already complete; [local_cleanup.md](local_cleanup.md) records device and port
closure and preservation of foreign containers.
