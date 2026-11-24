#!/usr/bin/env python3
"""Put the ELF cache into a tt-metal tree at any commit: copy elf_cache.py, hook _compile_kernel_part."""
import shutil
import sys
from pathlib import Path

tree, helper = Path(sys.argv[1]), Path(sys.argv[2])
helpers = tree / "tt_metal/tt-llk/tests/python_tests/helpers"
shutil.copy(helper, helpers / "elf_cache.py")
config = helpers / "test_config.py"
text = config.read_text()
if "from .elf_cache import compile_elf" in text:
    sys.exit(0)
old = """        if not cache:
            run_shell_command(compile_command, TestConfig.TESTS_WORKING_DIR, source)
"""
new = """        if not cache:
            if os.environ.get("LLK_ELF_CACHE") or os.environ.get("LLK_ELF_CACHE_STATS"):
                from .elf_cache import compile_elf

                compile_elf(compile_command, source, TestConfig.TESTS_WORKING_DIR)
                return
            run_shell_command(compile_command, TestConfig.TESTS_WORKING_DIR, source)
"""
assert text.count(old) == 1, "hook target not found"
config.write_text(text.replace(old, new))
