#!/usr/bin/env python3
"""
Automated Hardware Debugging Tool

A standalone script that injects debugging code into test functions, program factories,
and compute kernels to automatically find minimum failing configurations.

Usage:
    python automated_hardware_debugger.py --test-file tests/ttnn/unit_tests/operations/test_permute.py --function test_permute_5d_blocked
"""

import argparse
import ast
import os
import re
import sys
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import tempfile


@dataclass
class DebugConfig:
    """Configuration for debugging injection"""

    test_file: str
    function_name: str
    backup_dir: str
    nop_types: List[str] = None
    max_nops: int = 100
    iterations: int = 10
    skip_build: bool = False

    def __post_init__(self):
        if self.nop_types is None:
            self.nop_types = ["UNOPS", "MNOPS", "PNOPS"]


@dataclass
class FileModification:
    """Track modifications made to files"""

    file_path: str
    original_content: str
    modified_content: str
    backup_path: str


class CodeInjector:
    """Handles injection of debugging code into various file types"""

    def __init__(self, config: DebugConfig):
        self.config = config
        self.modifications: List[FileModification] = []

    def create_backup(self, file_path: str) -> str:
        """Create backup of original file"""
        backup_path = os.path.join(self.config.backup_dir, f"{os.path.basename(file_path)}.backup")
        shutil.copy2(file_path, backup_path)
        return backup_path

    def inject_test_debugging_loops(self, file_path: str, function_name: str) -> bool:
        """Inject debugging loops into test function"""
        print(f"🔧 Injecting debugging loops into {function_name} in {file_path}")

        with open(file_path, "r") as f:
            original_content = f.read()

        backup_path = self.create_backup(file_path)

        # Add required imports if not already present
        # modified_content = self._ensure_required_imports(original_content)
        modified_content = original_content

        # Parse the AST to find the function
        tree = ast.parse(original_content)

        # Find the target function
        target_function = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                target_function = node
                break

        if not target_function:
            print(f"❌ Function {function_name} not found in {file_path}")
            return False

        # Generate the debugging code injection
        debugging_injection = self._generate_debugging_injection()

        # Insert the debugging code - we need to replace the function body
        lines = modified_content.split("\n")

        # Find function definition line
        func_line_start = None
        for i, line in enumerate(lines):
            if f"def {function_name}(" in line:
                func_line_start = i
                break

        if func_line_start is None:
            print(f"❌ Could not locate function definition for {function_name}")
            return False

        # Find the end of function parameters and start of body
        func_body_start = None
        indent_level = 0
        for i in range(func_line_start, len(lines)):
            line = lines[i]
            if line.strip().endswith(":"):
                func_body_start = i + 1
                # Get indentation level of first line of function body
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    indent_level = len(next_line) - len(next_line.lstrip())
                break

        if func_body_start is None:
            print(f"❌ Could not locate function body start for {function_name}")
            return False

        # Find the end of the function
        func_end = None
        for i in range(func_body_start, len(lines)):
            line = lines[i]
            if line.strip() == "":
                continue
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= indent_level and line.strip() and not line.startswith(" " * indent_level):
                func_end = i
                break

        if func_end is None:
            func_end = len(lines)

        # Replace function body with debugging injection
        new_lines = lines[:func_body_start] + debugging_injection.split("\n") + lines[func_end:]

        modified_content = "\n".join(new_lines)

        # Write modified content
        with open(file_path, "w") as f:
            f.write(modified_content)

        # Track modification
        modification = FileModification(
            file_path=file_path,
            original_content=original_content,
            modified_content=modified_content,
            backup_path=backup_path,
        )
        self.modifications.append(modification)

        print(f"✅ Successfully injected debugging loops into {function_name}")
        return True

    def _ensure_required_imports(self, content: str) -> str:
        """Ensure required imports are present for the debugging code"""
        lines = content.split("\n")

        # Check if os import exists
        has_os_import = any("import os" in line for line in lines)
        # Check if json import exists
        has_json_import = any("import json" in line for line in lines)

        # Find where to insert imports (after existing imports)
        import_insertion_point = 0
        for i, line in enumerate(lines):
            if line.startswith("import ") or line.startswith("from "):
                import_insertion_point = i + 1
            elif line.strip() == "" and import_insertion_point > 0:
                continue  # Skip empty lines after imports
            elif import_insertion_point > 0:
                break  # Found first non-import, non-empty line

        # If no imports found, insert after first few comment lines
        if import_insertion_point == 0:
            for i, line in enumerate(lines):
                if line.startswith("#") or line.strip() == "":
                    continue
                else:
                    import_insertion_point = i
                    break

        # Insert missing imports
        imports_to_add = []
        if not has_os_import:
            imports_to_add.append("import os")
        if not has_json_import:
            imports_to_add.append("import json")

        if imports_to_add:
            # Insert imports at the insertion point
            new_lines = (
                lines[:import_insertion_point]
                + imports_to_add
                + [""]
                + lines[import_insertion_point:]  # Empty line after imports
            )
            content = "\n".join(new_lines)
            print(f"📦 Added missing imports: {', '.join(imports_to_add)}")

        return content

    def _generate_debugging_injection(self) -> str:
        """Generate the debugging loop injection code"""
        return f"""    device.disable_and_clear_program_cache()
    print("Shape: ", shape, "Perm: ", perm, "Memory config: ", memory_config, "Dtype: ", dtype)
    nop_types_sentence = "UNOPS MNOPS PNOPS"
    nop_types = nop_types_sentence.split()

    torch.manual_seed(520)
    input_a = random_torch_tensor(dtype, shape)
    torch_output = torch.permute(input_a, perm)

    # Automated debugging injection starts here
    min_config = {{}}
    all_results = []

    for is_risc in range(2):
        print("\\n🔄 RISCV ", is_risc, flush=True)
        os.environ["RISCV"] = str(is_risc)
        for core_nop in nop_types:
            print(f"\\n🧪 NOP TYPE {{core_nop}}", flush=True)
            my_it = {self.config.iterations}
            my_nop = {self.config.max_nops}
            min_nop = 0
            min_it = my_it

            for nops in range(my_nop):
                os.environ[core_nop] = str(nops)
                counter = 0
                print(f"  Testing {{core_nop}} nops={{nops}}: ", end="", flush=True)

                for i in range(my_it):
                    try:
                        tt_input = ttnn.from_torch(
                            input_a, device=device, layout=ttnn.ROW_MAJOR_LAYOUT,
                            dtype=dtype, memory_config=memory_config
                        )

                        tt_output = ttnn.permute(tt_input, perm)
                        tt_output = ttnn.to_torch(tt_output)

                        if torch.equal(torch_output, tt_output):
                            counter = counter + 1
                            print("✓", end="", flush=True)
                        else:
                            # Failure detected
                            print("✗", end="", flush=True)
                    except Exception as e:
                        # Exception counts as failure
                        print("E", end="", flush=True)

                    # Show progress every few iterations for long tests
                    if my_it > 20 and (i + 1) % 10 == 0:
                        print(f"[{{i+1}}/{{my_it}}]", end="", flush=True)

                failures = my_it - counter
                failure_rate = failures / my_it if my_it > 0 else 0
                print(f" → {{counter}}/{{my_it}} passed ({{failures}} failures)")  # Final result on new line

                if failures > 0:
                    result = {{
                        'nop_type': core_nop,
                        'risc_mode': is_risc,
                        'nop_count': nops,
                        'failures': failures,
                        'total_tests': my_it,
                        'failure_rate': failure_rate,
                        'shape': shape,
                        'perm': perm,
                        'memory_config': str(memory_config),
                        'dtype': str(dtype)
                    }}
                    all_results.append(result)
                    print(f"Nops {{nops}}: {{failures}}/{{my_it}} failures ({{failure_rate:.2%}})")

                if min_it > counter:
                    min_nop = nops
                    min_it = counter

            print(f"\\n✅ {{core_nop}} Summary: Min nops={{min_nop}}, Best counter={{min_it}}", flush=True)

    # Save results for analysis
    if all_results:
        results_file = f"debug_results_{{function_name}}_{{shape}}_{{perm}}.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"📊 Debug results saved to {{results_file}}")

        # Find optimal configuration
        best_result = min(all_results, key=lambda x: x['failure_rate'])
        print("🎯 Optimal debugging configuration:")
        print(f"  Nop Type: {{best_result['nop_type']}}")
        print(f"  RISC Mode: {{best_result['risc_mode']}}")
        print(f"  Nop Count: {{best_result['nop_count']}}")
        print(f"  Failure Rate: {{best_result['failure_rate']:.2%}}")
"""

    def find_program_factory_file(self, test_file: str) -> Optional[str]:
        """Find the program factory file using robust multi-strategy search"""
        print("🔍 Searching for program factory file...")

        # Determine operation type from test file
        operation_type = self._extract_operation_type(test_file)
        print(f"📍 Detected operation type: {operation_type}")

        # Multi-strategy search
        strategies = [
            self._find_files_with_git,
            self._find_files_with_find_command,
            self._find_files_with_content_search,
            self._find_files_direct_paths,
        ]

        for strategy_name, strategy_func in zip(
            ["Git-based search", "Find command", "Content search", "Direct paths"], strategies
        ):
            print(f"  🔄 Trying {strategy_name}...")
            try:
                result = strategy_func("program_factory", operation_type)
                if result:
                    print(f"✅ Found program factory: {result}")
                    return result
            except Exception as e:
                print(f"    ⚠️ {strategy_name} failed: {e}")
                continue

        print("❌ Could not find program factory file with any strategy")
        return None

    def _extract_operation_type(self, test_file: str) -> str:
        """Extract operation type from test file name and content"""
        file_name = os.path.basename(test_file).lower()

        # Extract from filename
        if "permute" in file_name:
            return "permute"
        elif "conv" in file_name:
            return "conv"
        elif "matmul" in file_name:
            return "matmul"
        elif "transpose" in file_name:
            return "transpose"

        # Try to extract from file content
        try:
            with open(test_file, "r") as f:
                content = f.read().lower()
                if "ttnn.permute" in content:
                    return "permute"
                elif "ttnn.conv" in content:
                    return "conv"
                elif "ttnn.matmul" in content:
                    return "matmul"
                elif "ttnn.transpose" in content:
                    return "transpose"
        except:
            pass

        return "unknown"

    def _find_files_with_git(self, file_type: str, operation: str) -> Optional[str]:
        """Use git to find files - most reliable in git repositories"""
        try:
            # Get all .cpp files from git
            result = subprocess.run(["git", "ls-files", "*.cpp"], capture_output=True, text=True, cwd=".", timeout=10)

            if result.returncode != 0:
                return None

            files = result.stdout.strip().split("\n")
            files = [f for f in files if f.strip()]  # Remove empty lines

            if file_type == "program_factory":
                candidates = []
                for f in files:
                    f_lower = f.lower()
                    if ("program_factory" in f_lower or "factory" in f_lower) and operation in f_lower:
                        candidates.append(f)

                # Prefer specific patterns
                for candidate in candidates:
                    if f"{operation}_rm_program_factory" in candidate.lower():
                        return candidate
                    elif f"{operation}_program_factory" in candidate.lower():
                        return candidate

                # Return first match if found
                if candidates:
                    return candidates[0]

            elif file_type == "compute_kernel":
                candidates = []
                for f in files:
                    f_lower = f.lower()
                    if (("kernel" in f_lower and "compute" in f_lower) or "transpose" in f_lower) and (
                        operation in f_lower or "transpose" in f_lower
                    ):
                        candidates.append(f)

                # Prefer specific patterns
                for candidate in candidates:
                    if "transpose_xw" in candidate.lower():
                        return candidate
                    elif f"{operation}" in candidate.lower() and "compute" in candidate.lower():
                        return candidate

                # Return first match if found
                if candidates:
                    return candidates[0]

            return None

        except Exception as e:
            print(f"    Git search failed: {e}")
            return None

    def _find_files_with_find_command(self, file_type: str, operation: str) -> Optional[str]:
        """Use system find command - works when git is not available"""
        try:
            if file_type == "program_factory":
                # Search for program factory files
                patterns = [f"*{operation}*program_factory*.cpp", f"*{operation}*factory*.cpp", "*program_factory*.cpp"]
            else:  # compute_kernel
                patterns = [f"*{operation}*compute*.cpp", "*transpose_xw*.cpp", "*transpose*.cpp"]

            for pattern in patterns:
                result = subprocess.run(
                    ["find", ".", "-name", pattern, "-type", "f"], capture_output=True, text=True, timeout=15
                )

                if result.returncode == 0 and result.stdout.strip():
                    files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]

                    # Filter and prioritize results
                    if file_type == "program_factory":
                        # Prefer rm_program_factory files
                        for f in files:
                            if "rm_program_factory" in f.lower():
                                return f
                        # Return first match
                        if files:
                            return files[0]
                    else:  # compute_kernel
                        # Prefer transpose_xw files
                        for f in files:
                            if "transpose_xw" in f.lower():
                                return f
                        # Return first match
                        if files:
                            return files[0]

            return None

        except Exception as e:
            print(f"    Find command failed: {e}")
            return None

    def _find_files_with_content_search(self, file_type: str, operation: str) -> Optional[str]:
        """Search files by content - finds files that actually implement the functionality"""
        try:
            if file_type == "program_factory":
                # Search for files containing ComputeConfig and environment variable patterns
                search_patterns = ["ComputeConfig", "CreateKernel", "program_factory"]
            else:  # compute_kernel
                search_patterns = ["MAIN", "get_compile_time_arg_val", "namespace NAMESPACE"]

            for pattern in search_patterns:
                try:
                    result = subprocess.run(
                        ["grep", "-r", "-l", pattern, "--include=*.cpp", "."],
                        capture_output=True,
                        text=True,
                        timeout=20,
                    )

                    if result.returncode == 0 and result.stdout.strip():
                        files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]

                        # Filter by operation and file type
                        candidates = []
                        for f in files:
                            f_lower = f.lower()
                            if file_type == "program_factory":
                                if "program_factory" in f_lower or "factory" in f_lower:
                                    if operation in f_lower or operation == "unknown":
                                        candidates.append(f)
                            else:  # compute_kernel
                                if ("kernel" in f_lower and "compute" in f_lower) or "transpose" in f_lower:
                                    if operation in f_lower or operation == "unknown" or "transpose" in f_lower:
                                        candidates.append(f)

                        # Return best candidate
                        if candidates:
                            # Prioritize by operation match
                            for candidate in candidates:
                                if operation in candidate.lower():
                                    return candidate
                            return candidates[0]

                except subprocess.TimeoutExpired:
                    print(f"    Content search for '{pattern}' timed out")
                    continue
                except Exception:
                    continue

            return None

        except Exception as e:
            print(f"    Content search failed: {e}")
            return None

    def _find_files_direct_paths(self, file_type: str, operation: str) -> Optional[str]:
        """Check common direct paths where files are typically located"""
        try:
            base_paths = [
                "ttnn/cpp/ttnn/operations",
                "ttnn/operations",
                "tt_eager/tt_dnn/op_library",
                "tt_metal/programming_examples",
            ]

            if file_type == "program_factory":
                file_patterns = [
                    f"**/{operation}*program_factory*.cpp",
                    f"**/device/{operation}*program_factory*.cpp",
                    f"**/device/*program_factory*.cpp",
                    "**/device/*factory*.cpp",
                ]
            else:  # compute_kernel
                file_patterns = [
                    f"**/kernels/compute/{operation}*.cpp",
                    "**/kernels/compute/transpose*.cpp",
                    "**/kernels/compute/*.cpp",
                ]

            for base_path in base_paths:
                if not os.path.exists(base_path):
                    continue

                for pattern in file_patterns:
                    full_pattern = os.path.join(base_path, pattern.lstrip("**/"))
                    try:
                        matches = list(Path(base_path).rglob(pattern.lstrip("**/")))
                        if matches:
                            # Prioritize matches
                            for match in matches:
                                match_str = str(match).lower()
                                if file_type == "program_factory":
                                    if "rm_program_factory" in match_str:
                                        return str(match)
                                else:  # compute_kernel
                                    if "transpose_xw" in match_str:
                                        return str(match)

                            # Return first match
                            return str(matches[0])
                    except Exception:
                        continue

            return None

        except Exception as e:
            print(f"    Direct path search failed: {e}")
            return None

    def inject_program_factory_modifications(self, factory_file: str) -> bool:
        """Inject environment variable handling into program factory"""
        print(f"🔧 Injecting program factory modifications into {factory_file}")

        with open(factory_file, "r") as f:
            original_content = f.read()

        backup_path = self.create_backup(factory_file)

        # Find the location to inject the environment variable handling
        # Look for ComputeConfig creation
        compute_config_pattern = r"tt::tt_metal::ComputeConfig\s*\{"

        if not re.search(compute_config_pattern, original_content):
            print("❌ Could not find ComputeConfig in program factory")
            return False

        # Add the environment variable handling code
        env_var_code = """
    std::map<std::string, std::string> compute_defines;
    compute_defines["UNOPS"] = std::to_string(std::getenv("UNOPS") ? std::stoi(std::getenv("UNOPS")) : 0);
    compute_defines["MNOPS"] = std::to_string(std::getenv("MNOPS") ? std::stoi(std::getenv("MNOPS")) : 0);
    compute_defines["PNOPS"] = std::to_string(std::getenv("PNOPS") ? std::stoi(std::getenv("PNOPS")) : 0);
    compute_defines["RISCV"] = std::to_string(std::getenv("RISCV") ? std::stoi(std::getenv("RISCV")) : 0);
    for (const auto& pair : compute_defines) {
        // std::cout << "Key: " << pair.first << ", Value: " << pair.second << std::endl;
    }"""

        modified_content = original_content

        # Only inject if not already present
        if 'compute_defines["UNOPS"]' not in modified_content:
            # Insert environment variable code before ComputeConfig
            modified_content = re.sub(
                r"(\s+)(std::vector<uint32_t>\s+compute_kernel_args\s*=)",
                r"\1" + env_var_code + r"\n\1\2",
                modified_content,
            )

        # Add .defines = compute_defines to ComputeConfig (only if not already present)
        if ".defines = compute_defines" not in modified_content:
            modified_content = re.sub(
                r"(tt::tt_metal::ComputeConfig\s*\{[^}]*)(\.fp32_dest_acc_en\s*=\s*[^,}]+,?\s*\.compile_args\s*=\s*[^,}]+),?\s*\}",
                r"\1\2, .defines = compute_defines}",
                modified_content,
            )

        # Add necessary includes for environment variable handling
        includes_to_add = []
        if "#include <iostream>" not in modified_content:
            includes_to_add.append("#include <iostream>")
        if "#include <cstdlib>" not in modified_content:
            includes_to_add.append("#include <cstdlib>")  # for std::getenv
        if "#include <string>" not in modified_content:
            includes_to_add.append("#include <string>")  # for std::stoi, std::to_string
        if "#include <map>" not in modified_content:
            includes_to_add.append("#include <map>")  # for std::map

        if includes_to_add:
            # Find the last include and insert after it
            last_include_pattern = r"(#include\s+<[^>]+>\s*\n)(?!.*#include\s+<)"
            for include in includes_to_add:
                modified_content = re.sub(last_include_pattern, r"\1" + include + "\n", modified_content, count=1)

        # Write modified content
        with open(factory_file, "w") as f:
            f.write(modified_content)

        # Track modification
        modification = FileModification(
            file_path=factory_file,
            original_content=original_content,
            modified_content=modified_content,
            backup_path=backup_path,
        )
        self.modifications.append(modification)

        print("✅ Successfully injected program factory modifications")
        return True

    def find_compute_kernel_file(self, test_file: str) -> Optional[str]:
        """Find the compute kernel file using robust multi-strategy search"""
        print("🔍 Searching for compute kernel file...")

        # Determine operation type from test file
        operation_type = self._extract_operation_type(test_file)
        print(f"📍 Detected operation type: {operation_type}")

        # Multi-strategy search (reusing the same robust methods)
        strategies = [
            self._find_files_with_git,
            self._find_files_with_find_command,
            self._find_files_with_content_search,
            self._find_files_direct_paths,
        ]

        for strategy_name, strategy_func in zip(
            ["Git-based search", "Find command", "Content search", "Direct paths"], strategies
        ):
            print(f"  🔄 Trying {strategy_name}...")
            try:
                result = strategy_func("compute_kernel", operation_type)
                if result:
                    print(f"✅ Found compute kernel: {result}")
                    return result
            except Exception as e:
                print(f"    ⚠️ {strategy_name} failed: {e}")
                continue

        print("❌ Could not find compute kernel file with any strategy")
        return None

    def inject_compute_kernel_modifications(self, kernel_file: str) -> bool:
        """Inject NOP functions and calls into compute kernel"""
        print(f"🔧 Injecting compute kernel modifications into {kernel_file}")

        with open(kernel_file, "r") as f:
            original_content = f.read()

        backup_path = self.create_backup(kernel_file)

        # Add debug includes if not present
        debug_includes = """#include "debug/dprint.h"
#include "debug/dprint_pages.h"
#include "debug/dprint_tensix.h"
"""

        # Add NOP functions
        nop_functions = """
template <const int n, const int riscv>
inline void add_nops() {
    DPRINT << "RISCV " << riscv << " NOPS " << n << ENDL();

    for (int i = 0; i < n; i++) {
        if constexpr (riscv) {
            asm("nop");
        } else {
            TTI_NOP;
        }
    }
}

template <const int U, const int M, const int P, const int R>
inline void add_trisc_nops() {
    DPRINT << "U " << (uint32_t)U << " M " << (uint32_t)M << " P " << (uint32_t)P << ENDL();
    if constexpr (U) {
        UNPACK((add_nops<U, R>()));
    }

    if constexpr (M) {
        MATH((add_nops<M, R>()));
    }

    if constexpr (P) {
        PACK((add_nops<P, R>()));
    }
}
"""

        modified_content = original_content

        # Add debug includes after existing includes (only if not already present)
        if "debug/dprint.h" not in modified_content:
            include_pattern = r'(#include\s+[<"][^>"]+[>"]\s*\n)'
            includes = re.findall(include_pattern, modified_content)
            if includes:
                last_include = includes[-1]
                modified_content = modified_content.replace(last_include, last_include + debug_includes)

        # Add NOP functions before namespace (only if not already present)
        if "add_nops()" not in modified_content:
            namespace_pattern = r"(namespace\s+\w+\s*\{)"
            if re.search(namespace_pattern, modified_content):
                modified_content = re.sub(namespace_pattern, nop_functions + r"\n\1", modified_content)

        # Find the main loop and inject NOP call
        # Look for common patterns in compute kernels
        main_loop_patterns = [
            r"(for\s*\(\s*uint32_t\s+n\s*=\s*0\s*;\s*n\s*<\s*num_blocks\s*;\s*n\+\+\s*\)\s*\{)",
            r"(for\s*\(\s*uint32_t\s+\w+\s*=\s*0\s*;\s*\w+\s*<\s*\w+\s*;\s*\w+\+\+\s*\)\s*\{)",
        ]

        nop_injection = "        add_trisc_nops<UNOPS, MNOPS, PNOPS, RISCV>();\n"

        # Only inject NOP call if not already present
        if "add_trisc_nops<UNOPS, MNOPS, PNOPS, RISCV>();" not in modified_content:
            for pattern in main_loop_patterns:
                if re.search(pattern, modified_content):
                    modified_content = re.sub(pattern, r"\1\n" + nop_injection, modified_content)
                    break

        # Write modified content
        with open(kernel_file, "w") as f:
            f.write(modified_content)

        # Track modification
        modification = FileModification(
            file_path=kernel_file,
            original_content=original_content,
            modified_content=modified_content,
            backup_path=backup_path,
        )
        self.modifications.append(modification)

        print("✅ Successfully injected compute kernel modifications")
        return True

    def restore_all_modifications(self):
        """Restore all modified files from backups and clean up backup files"""
        print("🔄 Restoring all modified files...")

        backup_files_to_remove = []

        for modification in self.modifications:
            try:
                shutil.copy2(modification.backup_path, modification.file_path)
                print(f"✅ Restored {modification.file_path}")
                # Track backup file for cleanup
                backup_files_to_remove.append(modification.backup_path)
            except Exception as e:
                print(f"❌ Failed to restore {modification.file_path}: {e}")

        # Clean up backup files
        if backup_files_to_remove:
            print("🧹 Cleaning up backup files...")
            for backup_file in backup_files_to_remove:
                try:
                    os.remove(backup_file)
                    print(f"🗑️ Removed backup: {os.path.basename(backup_file)}")
                except Exception as e:
                    print(f"⚠️ Could not remove backup {backup_file}: {e}")

            # Try to remove backup directory if it's empty
            try:
                backup_dir = os.path.dirname(backup_files_to_remove[0]) if backup_files_to_remove else None
                if backup_dir and os.path.exists(backup_dir):
                    # Check if directory is empty
                    if not os.listdir(backup_dir):
                        os.rmdir(backup_dir)
                        print(f"🗑️ Removed empty backup directory: {os.path.basename(backup_dir)}")
                    else:
                        print(f"📁 Backup directory contains other files, keeping: {os.path.basename(backup_dir)}")
            except Exception as e:
                print(f"⚠️ Could not remove backup directory: {e}")

        print("🔄 All files restored and backups cleaned up")


class AutomatedHardwareDebugger:
    """Main class for the automated hardware debugging tool"""

    def __init__(self, config: DebugConfig):
        self.config = config
        self.injector = CodeInjector(config)

        # Create backup directory
        os.makedirs(config.backup_dir, exist_ok=True)

        # Create modified files directory
        self.modified_files_dir = "automated_hardware_debugger/modified_files_snapshot"
        os.makedirs(self.modified_files_dir, exist_ok=True)

    def save_modified_files_snapshot(self):
        """Save copies of all modified files for inspection"""
        if not self.injector.modifications:
            return

        print("📸 Saving snapshot of modified files...")

        for modification in self.injector.modifications:
            try:
                # Create filename for modified version
                file_basename = os.path.basename(modification.file_path)
                modified_filename = f"modified_{file_basename}"
                modified_path = os.path.join(self.modified_files_dir, modified_filename)

                # Copy current (modified) file to snapshot directory
                shutil.copy2(modification.file_path, modified_path)
                print(f"📸 Saved modified: {file_basename} -> {modified_filename}")
            except Exception as e:
                print(f"⚠️ Could not save modified file {modification.file_path}: {e}")

        print(f"📁 Modified files saved in: {self.modified_files_dir}/")
        print("   These show exactly what code was injected and executed.")

    def run_debugging_session(self) -> Dict[str, Any]:
        """Run the complete debugging session"""
        print("🚀 Starting Automated Hardware Debugging Session")
        print("=" * 80)

        try:
            # Step 1: Inject debugging loops into test function
            success = self.injector.inject_test_debugging_loops(self.config.test_file, self.config.function_name)
            if not success:
                return {"error": "Failed to inject test debugging loops"}

            # Step 2: Find and modify program factory
            program_factory = self.injector.find_program_factory_file(self.config.test_file)
            if program_factory:
                success = self.injector.inject_program_factory_modifications(program_factory)
                if not success:
                    print("⚠️ Warning: Failed to modify program factory")
            else:
                print("⚠️ Warning: Could not find program factory file")

            # Step 3: Find and modify compute kernel
            compute_kernel = self.injector.find_compute_kernel_file(self.config.test_file)
            if compute_kernel:
                success = self.injector.inject_compute_kernel_modifications(compute_kernel)
                if not success:
                    print("⚠️ Warning: Failed to modify compute kernel")
            else:
                print("⚠️ Warning: Could not find compute kernel file")

            # Step 4: Build the project (required after C++ modifications)
            if self.config.skip_build:
                print("\n⏭️ Skipping project build (--skip-build flag provided)")
                print("⚠️ Warning: C++ modifications may not take effect without rebuilding")
            else:
                print("\n🔨 Building project with modified files...")
                build_success = self._build_project()
                if not build_success:
                    print("⚠️ Warning: Build failed, but continuing with test execution")

            # Step 5: Run the modified test
            print("\n🧪 Running modified test...")
            result = self._run_test()

            # Step 6: Analyze results
            analysis = self._analyze_results(result)

            return {
                "success": True,
                "test_result": result,
                "analysis": analysis,
                "modified_files": [mod.file_path for mod in self.injector.modifications],
            }

        except Exception as e:
            print(f"❌ Error during debugging session: {e}")
            return {"error": str(e)}

        finally:
            # Save snapshot of modified files before restoration
            self.save_modified_files_snapshot()

            # Always restore original files
            self.injector.restore_all_modifications()

    def _build_project(self) -> bool:
        """Build the project after C++ modifications"""
        print("⚙️ Setting environment and building project...")

        build_command = ["bash", "-c", "export TT_METAL_ENV=dev && ./build_metal.sh --enable-profiler --build-tests"]

        try:
            # Use Popen to stream output in real-time
            process = subprocess.Popen(
                build_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr with stdout
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True,
            )

            stdout_lines = []
            start_time = time.time()
            timeout = 600  # 10 minutes

            # Stream output in real-time
            while True:
                # Check for timeout
                if time.time() - start_time > timeout:
                    process.terminate()
                    process.wait()
                    print("❌ Build timed out after 10 minutes")
                    return False

                output = process.stdout.readline()
                if output == "" and process.poll() is not None:
                    break
                if output:
                    print(output.strip())  # Print to terminal immediately
                    stdout_lines.append(output)

            returncode = process.poll()

            if returncode == 0:
                print("✅ Project built successfully")
                return True
            else:
                print(f"❌ Build failed with return code {returncode}")
                return False

        except Exception as e:
            print(f"❌ Build failed with exception: {e}")
            return False

    def _run_test(self) -> Dict[str, Any]:
        """Run the modified test and capture results"""
        test_command = [
            "python3",
            "-m",
            "pytest",
            f"{self.config.test_file}::{self.config.function_name}",
            "-v",
            "-s",
            "--tb=short",
            "--capture=no",
            "--log-cli-level=INFO",
        ]

        try:
            # Use Popen to stream output in real-time
            process = subprocess.Popen(
                test_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr with stdout
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True,
            )

            stdout_lines = []
            start_time = time.time()
            timeout = 300  # 5 minutes

            # Stream output in real-time while capturing for analysis
            while True:
                # Check for timeout
                if time.time() - start_time > timeout:
                    process.terminate()
                    process.wait()
                    return {"error": "Test execution timed out after 5 minutes", "command": " ".join(test_command)}

                output = process.stdout.readline()
                if output == "" and process.poll() is not None:
                    break
                if output:
                    print(output.strip())  # Print to terminal immediately
                    stdout_lines.append(output)

            returncode = process.poll()
            captured_stdout = "".join(stdout_lines)

            return {
                "returncode": returncode,
                "stdout": captured_stdout,
                "stderr": "",  # Already merged with stdout
                "command": " ".join(test_command),
            }

        except Exception as e:
            return {"error": f"Failed to run test: {e}", "command": " ".join(test_command)}

    def _analyze_results(self, test_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the test results and extract debugging information"""
        if "error" in test_result:
            return {"error": test_result["error"]}

        analysis = {"test_passed": test_result["returncode"] == 0, "debug_results_files": [], "optimal_configs": []}

        # Look for generated JSON result files
        for file_path in Path(".").glob("debug_results_*.json"):
            try:
                with open(file_path, "r") as f:
                    results = json.load(f)

                analysis["debug_results_files"].append(str(file_path))

                # Find optimal configuration
                if results:
                    optimal = min(results, key=lambda x: x.get("failure_rate", 1.0))
                    analysis["optimal_configs"].append({"file": str(file_path), "config": optimal})

                # Clean up result file
                os.remove(file_path)

            except Exception as e:
                print(f"⚠️ Warning: Could not analyze {file_path}: {e}")

        return analysis


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Automated Hardware Debugging Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python automated_hardware_debugger.py --test-file tests/ttnn/unit_tests/operations/test_permute.py --function test_permute_5d_blocked

  python automated_hardware_debugger.py --test-file tests/ttnn/unit_tests/operations/test_permute.py --function test_permute_5d_blocked --max-nops 50 --iterations 5

  python automated_hardware_debugger.py --test-file tests/ttnn/unit_tests/operations/test_permute.py --function test_permute_5d_blocked --skip-build
        """,
    )

    parser.add_argument("--test-file", required=True, help="Path to the test file containing the function to debug")

    parser.add_argument("--function", required=True, help="Name of the test function to debug")

    parser.add_argument("--max-nops", type=int, default=100, help="Maximum number of NOPs to test (default: 100)")

    parser.add_argument(
        "--iterations", type=int, default=10, help="Number of iterations per configuration (default: 10)"
    )

    parser.add_argument(
        "--backup-dir", default="./debug_backups", help="Directory to store backup files (default: ./debug_backups)"
    )

    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip the project build step (useful for testing, but may cause issues if C++ files were modified)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.test_file):
        print(f"❌ Error: Test file {args.test_file} does not exist")
        sys.exit(1)

    # Create configuration
    config = DebugConfig(
        test_file=args.test_file,
        function_name=args.function,
        backup_dir=args.backup_dir,
        max_nops=args.max_nops,
        iterations=args.iterations,
        skip_build=args.skip_build,
    )

    # Run debugging session
    debugger = AutomatedHardwareDebugger(config)
    result = debugger.run_debugging_session()

    # Print results
    print("\n" + "=" * 80)
    print("DEBUGGING SESSION RESULTS")
    print("=" * 80)

    if "error" in result:
        print(f"❌ Debugging session failed: {result['error']}")
        sys.exit(1)

    print("✅ Debugging session completed successfully!")

    if result["analysis"].get("optimal_configs"):
        print("\n🎯 OPTIMAL DEBUGGING CONFIGURATIONS:")
        for i, config_info in enumerate(result["analysis"]["optimal_configs"], 1):
            config_data = config_info["config"]
            print(f"\nConfiguration {i}:")
            print(f"  Shape: {config_data.get('shape', 'N/A')}")
            print(f"  Permutation: {config_data.get('perm', 'N/A')}")
            print(f"  Memory Config: {config_data.get('memory_config', 'N/A')}")
            print(f"  Data Type: {config_data.get('dtype', 'N/A')}")
            print(f"  Nop Type: {config_data.get('nop_type', 'N/A')}")
            print(f"  RISC Mode: {config_data.get('risc_mode', 'N/A')}")
            print(f"  Nop Count: {config_data.get('nop_count', 'N/A')}")
            print(f"  Failure Rate: {config_data.get('failure_rate', 0):.2%}")
    else:
        print("\n📊 No failing configurations found - all tests passed!")

    print(f"\n📁 Modified files: {len(result['modified_files'])}")
    for file_path in result["modified_files"]:
        print(f"  • {file_path}")

    print("\n🔄 All files have been restored to their original state.")


if __name__ == "__main__":
    main()
