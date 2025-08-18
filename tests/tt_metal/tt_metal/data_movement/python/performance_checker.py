# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from loguru import logger  # type: ignore


class PerformanceChecker:
    def __init__(self, dm_stats, arch="blackhole", verbose=False, test_bounds=None, test_id_to_name=None):
        self.dm_stats = dm_stats
        self.arch = arch
        self.verbose = verbose
        self.test_bounds = test_bounds
        self.test_id_to_name = test_id_to_name
        self.results_bounds = self._calculate_results_bounds()

    def _calculate_results_bounds(self):
        results_bounds = {}

        for riscv in self.dm_stats.keys():
            for run in self.dm_stats[riscv]["analysis"]["series"]:
                run_host_id = run["duration_type"][0]["run_host_id"]
                test_id = self.dm_stats[riscv]["attributes"][run_host_id]["Test id"]

                if test_id not in results_bounds.keys():
                    results_bounds[test_id] = {}
                if riscv not in results_bounds[test_id].keys():
                    results_bounds[test_id][riscv] = {
                        "latency": 0,
                        "bandwidth": float("inf"),
                    }

                cycles = run["duration_cycles"]

                results_bounds[test_id][riscv]["latency"] = max(results_bounds[test_id][riscv]["latency"], cycles)

                attributes = self.dm_stats[riscv]["attributes"][run_host_id]
                bandwidth = attributes["Number of transactions"] * attributes["Transaction size in bytes"] / cycles

                results_bounds[test_id][riscv]["bandwidth"] = min(
                    results_bounds[test_id][riscv]["bandwidth"], bandwidth
                )

        return results_bounds

    def _log_perf_results(self, test_id, bounds):
        logger.info("")
        test_name = self.test_id_to_name.get(test_id, "Unknown Test")
        logger.info(f"Perf results for test id: {test_id} ({test_name})")

        logger.info(f"Latency")
        for riscv in bounds.keys():
            if bounds[riscv]["latency"] != float("inf"):
                logger.info(f"  {riscv}: {bounds[riscv]['latency']} cycles")

        logger.info(f"Bandwidth")
        for riscv in bounds.keys():
            if bounds[riscv]["bandwidth"] != float("inf"):
                logger.info(f"  {riscv}: {bounds[riscv]['bandwidth']} Bytes/cycle")

    def _check_test_performance(self, test_id, bounds):
        if self.test_bounds is None or self.arch not in self.test_bounds or test_id not in self.test_bounds[self.arch]:
            logger.warning(f"Test id {test_id} not found in {self.arch} test bounds.")
            return

        for riscv, riscv_bounds in bounds.items():
            if riscv not in self.test_bounds[self.arch][test_id]:
                continue

            expected_bw = self.test_bounds[self.arch][test_id][riscv]["bandwidth"]
            actual_bw = riscv_bounds["bandwidth"]
            bw_within_bounds = expected_bw <= actual_bw

            if self.verbose:
                if not bw_within_bounds:
                    logger.warning(
                        f"{riscv} bandwidth not within perf bounds. Expected >= {expected_bw}, got {actual_bw}"
                    )
                else:
                    logger.info(f"{riscv} bandwidth within perf bounds.")

            assert (
                bw_within_bounds
            ), f"{riscv} for test {self.test_id_to_name.get(test_id, 'Unknown Test')} ({test_id}) bandwidth not within perf bounds. Expected >= {expected_bw}, got {actual_bw}"

    def run(self):
        # Performance checks per test
        for test_id, bounds in self.results_bounds.items():
            if self.verbose:
                self._log_perf_results(test_id, bounds)

            self._check_test_performance(test_id, bounds)
