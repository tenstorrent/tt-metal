# Harbor source freeze defect diagnosis

Failing contract: an immutable model release configuration must select the same client harness source in dry run and release. Immutable TTI SHAs currently select a moving Harbor branch, so they do not prove that contract.

Causal chain verified in workflows/workflow_venvs.py: HARBOR_REF is the literal tt-inference-server branch; setup_evals_agentic ignores model_spec; checkout_pinned_repo always fetches the ref and checks out FETCH_HEAD, including on cached runners. VenvConfig.setup still invokes setup on an existing environment. Therefore branch advancement between runs changes Harbor source even when the TTI/model commits are identical. The pinned mini2.2.8 agent version does not control Harbor host source or Terminus code.

Competing explanation rejected: cached venvs freeze Harbor. The unconditional fetch/forced detach in checkout_pinned_repo disproves it. Environment/package overrides cannot select the local editable repository commit; no stock metadata field was read for that purpose.

Discriminating host test: request a full model-scoped SHA and assert exact ref passed to the unchanged checkout helper; reject branch names before checkout; compare actual git HEAD to requested SHA before install. Source and mocked host tests suffice to validate ref forwarding/failure behavior. Actual CI dependency identity remains to be collected before a model pin value is selected.
