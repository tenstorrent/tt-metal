# Multihost Test Suite

This directory contains **all integration and fault‑tolerance tests** for the distributed runtime.  Tests are organised to make it clear **what host topology they expect** and **how they should be launched**.

---

## Folder layout

| Sub‑folder                   | Purpose                                                                                                                                                                                                                  | Typical launcher                                       |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------ |
| **common/**                  | Utility code shared by all tests.  Includes:<br>• **sync\_helpers.hpp** – rank‑synchronised GoogleTest wrappers<br>• tooling helpers and mock objects                                                                    | *Header‑only* (no launcher)                            |
| **fault\_tolerance\_tests/** | Tests that *deliberately* kill ranks/hosts and validate recovery logic (ULFM, communicator shrink, respawn, device hot‑plug).<br><br>Each test is compiled into its own binary so it can be invoked in a fresh `mpirun`. | `mpirun -np <N> ./ft_case_X`                           |
| **multi\_hosts\_tests/**     | End‑to‑end tests that require **one process per host** and access to hardware devices (mesh, fabric, RDMA). They verify cross‑host barriers, collective latency, and multi‑device kernels.                               | `mpirun --hostfile hosts.txt -ppn 1 ./multihost_suite` |
| **single\_host\_mp\_tests/** | Fast checks that need **multiple ranks** but run on a single box (no devices). Used to validate the distributed context, serializers, span wrappers, etc.                                                                | `mpirun -np 8 ./single_host_suite`                     |


---

## Building

```bash
TODO (if we need something special)
```

Targets created:

* `multi_hosts_suite`
* `single_host_suite`
* One binary per fault‑tolerance test, e.g. `ft_comm_recover`, `ft_rank_restart`, …

---

## Running tests locally

### 1 · Single‑host, multi‑process

```bash
mpirun -np 8 ./single_host_suite --gtest_output=xml:results_single.xml
```

### 2 · Multi‑host functional

```bash
mpirun --hostfile hosts.txt -ppn 1 ./multi_hosts_suite \
       --gtest_filter=DeviceMesh.*
```

*Tip:* pass `--bind-to none` if you oversubscribe CPUs in CI.

### 3 · Fault‑tolerance scenarios

Each binary is launched **independently** so that a forced abort doesn’t kill the rest of the suite:

```bash
mpirun -np 4 ./ft_suite --gtest_filter=FaultTolerance.comm_recover
mpirun -np 8 ./ft_suite --gtest_filter=FaultTolerance.rank_restart
```

Scripts in `fault_tolerance_tests/run_all.sh` automate the sequence and consolidate XML reports.


---

## Adding a new test

1. Pick the right sub‑folder based on the topology you need.
2. Create a `*_test.cpp` file; include `<gtest/gtest.h>` **and** `common/sync_helpers.hpp`.
3. If it belongs to *fault\_tolerance\_tests*, add a new executable in the local `CMakeLists.txt` – one binary per disruption scenario.
4. Ensure any device allocation is wrapped in `ASSERT_EQ_ALL_RANKS` so missing devices cause a collective fail.

---



---

## Troubleshooting


Questions or patches: open an issue or ping `@dmakoviichuk-tt`
