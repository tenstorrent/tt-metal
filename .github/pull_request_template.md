### Ticket
Link to Github Issue

### Problem description
Provide context for the problem.

### What's changed
Describe the approach used to solve the problem.
Summarize the changes made and its impact.

### Checklist
- [ ] [All post commit](https://github.com/tenstorrent/tt-metal/actions/workflows/all-post-commit-workflows.yaml) CI passes
- [ ] [Blackhole Post commit](https://github.com/tenstorrent/tt-metal/actions/workflows/blackhole-post-commit.yaml) CI with demo tests passes (if applicable)
- [ ] [Model regression](https://github.com/tenstorrent/tt-metal/actions/workflows/perf-models.yaml) CI passes (if applicable)
- [ ] [Device performance regression](https://github.com/tenstorrent/tt-metal/actions/workflows/perf-device-models.yaml) CI passes (if applicable)
- [ ] **(For models and ops writers)** Full [single-card suite](https://github.com/tenstorrent/tt-metal/blob/main/models/MODEL_ADD.md#a-recommended-dev-flow-on-github-for-adding-new-models) CI passes (if applicable)
- [ ] New/Existing tests provide coverage for changes
