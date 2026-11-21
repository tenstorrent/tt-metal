"""
Static checker for Tensix compute-kernel API ordering rules.
Enforces correct pairing and ordering of init -> compute -> uninit / reconfig calls.
"""
import re
from typing import List, Dict, Any

class KernelApiOrderChecker:
    def __init__(self):
        self.init_calls = ["mm_init", "binary_init", "unary_init", "tilize_init", "untilize_init"]
        self.uninit_calls = ["mm_uninit", "binary_uninit", "unary_uninit", "tilize_uninit", "untilize_uninit"]
        self.reconfig_calls = ["mm_reconfig", "reconfig_data_format"]

    def check_source(self, code: str) -> List[Dict[str, Any]]:
        violations = []
        lines = code.splitlines()
        state = "UNINIT"
        current_init = None

        for idx, line in enumerate(lines, 1):
            clean = line.strip()
            if clean.startswith("//") or clean.startswith("/*"):
                continue

            # Check init
            for icall in self.init_calls:
                if icall in clean:
                    if state == "INIT":
                        violations.append({
                            "line": idx,
                            "rule": "DOUBLE_INIT_WITHOUT_UNINIT",
                            "message": f"Illegal double init '{icall}' without uninit of '{current_init}'"
                        })
                    state = "INIT"
                    current_init = icall

            # Check uninit
            for ucall in self.uninit_calls:
                if ucall in clean:
                    if state == "UNINIT":
                        violations.append({
                            "line": idx,
                            "rule": "UNINIT_WITHOUT_INIT",
                            "message": f"Illegal uninit '{ucall}' called when engine is already in UNINIT state"
                        })
                    state = "UNINIT"
                    current_init = None

            # Check compute operations when uninit
            if "tile_regs_acquire();" in clean or "matmul_tiles(" in clean:
                if state == "UNINIT":
                    violations.append({
                        "line": idx,
                        "rule": "COMPUTE_BEFORE_INIT",
                        "message": "Compute operation invoked before hardware initialization call"
                    })

        return violations
