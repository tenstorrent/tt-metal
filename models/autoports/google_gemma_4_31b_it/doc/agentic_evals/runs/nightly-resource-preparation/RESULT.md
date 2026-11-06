# Resource facts

Reused setup/task_resources.json and verified all15 task.toml hashes against pinned metadata checkouts (SWE86723674f04e4209ac479d0fb75d9d9f44b4377e, Terminal12b7565679ce5006230b820666353673a133db93). No duplicate inventory. All inventoried environment fields match; Terminal definitions additionally contain empty mcp_servers and env fields. result.json gives exact source paths and cohort arithmetic.

SWE nightly5 declares50GiB total storage; simultaneous C2 needs2CPUs,8GiB RAM,20GiB storage. Terminal nightly10 declares100GiB storage; worst C2 needs2CPUs,6GiB RAM,20GiB storage (dna-insert4GiB plus one2GiB task). These totals exclude serving/host overhead and retained images/build layers.

Pinned Harbor e76c762fca5267dfb36e1f900f0a7724ed72e735 models/task/config.py:449 defaults omitted cpus/memory_mb/storage_mb to None, not numeric allowances. No selected task omits them. Its legacy size conversion at506 maps G to1024MB. Docker implementation environments/docker/docker.py:290 passes CPU/memory into compose; resource capabilities at327 expose CPU/memory limits. Storage is declared accounting, not an enforced Docker quota. models/trial/config.py:224 defaults enforcement to AUTO. No blanket overrides added.

No existing task admission reserve was found in stage preflight/inputs or stock llm_module/agentic/harbor.py. scripts/build_docker_images.py has20GiB build reserve, which is not established as task admission reserve. Do not silently substitute it. Historical runner reports2045GB (Terminal retry) and1000GB (SWE startup) exceed declared totals alone, but do not establish current or future runner headroom. Full admission remains unknown until the actual selected runner reports Docker filesystem free capacity and applicable reserve/overhead. No stage-owned admission hook created. Nominal31 Docker networks exceeds C2, but occupied networks remain runner-specific.

Device cost0; speed saving0. Validity preparation only, no score or task-selection changes. No solutions read, publication edits, CI dispatch, or device work.
