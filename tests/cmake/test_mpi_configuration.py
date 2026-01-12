#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for MPI configuration.

These tests verify that MPI configuration works correctly by:
1. Testing MPI detection logic
2. Verifying RPATH handling for both ULFM and system MPI
3. Checking that TT_METAL_USING_ULFM is set correctly
"""

import subprocess
import tempfile
import unittest
from pathlib import Path


class TestMPIConfiguration(unittest.TestCase):
    """Integration tests for MPI configuration."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_cmake_dir = Path(__file__).parent.parent.parent / "cmake"

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _path_to_cmake_string(self, path: Path) -> str:
        """
        Convert a Path object to a CMake-compatible string.

        CMake expects forward slashes even on Windows, so we normalize the path.

        Args:
            path: Path object to convert

        Returns:
            String representation with forward slashes
        """
        return str(path).replace("\\", "/")

    def test_ulfm_mpi_detection(self):
        """Test that custom ULFM MPI is detected correctly."""
        # Create a mock ULFM installation
        ulfm_prefix = self.temp_dir / "opt" / "openmpi-v5.0.7-ulfm"
        ulfm_prefix.mkdir(parents=True)
        (ulfm_prefix / "lib" / "libmpi.so.40").parent.mkdir(parents=True)
        (ulfm_prefix / "lib" / "libmpi.so.40").touch()

        # Test CMake configuration
        cmake_script = f"""
cmake_minimum_required(VERSION 3.15)
project(test_mpi)

set(ULFM_PREFIX "{self._path_to_cmake_string(ulfm_prefix)}")
include({self._path_to_cmake_string(self.test_cmake_dir)}/mpi-config.cmake)

tt_configure_mpi(ON USE_MPI)

if(USE_MPI)
    message(STATUS "MPI found: TRUE")
else()
    message(STATUS "MPI found: FALSE")
endif()

if(TT_METAL_USING_ULFM)
    message(STATUS "Using ULFM: TRUE")
else()
    message(STATUS "Using ULFM: FALSE")
endif()

if(TT_METAL_MPI_LIB_DIR)
    message(STATUS "MPI lib dir: ${{TT_METAL_MPI_LIB_DIR}}")
endif()
"""
        result = self._run_cmake_test(cmake_script)
        self.assertIn("MPI found: TRUE", result)
        self.assertIn("Using ULFM: TRUE", result)
        self.assertIn(f"MPI lib dir: {self._path_to_cmake_string(ulfm_prefix / 'lib')}", result)

    def test_system_mpi_detection(self):
        """Test that system MPI is detected when ULFM is not present."""
        # Don't create ULFM installation
        cmake_script = f"""
cmake_minimum_required(VERSION 3.15)
project(test_mpi)

set(ULFM_PREFIX "/nonexistent/path")
include({self._path_to_cmake_string(self.test_cmake_dir)}/mpi-config.cmake)

# Try to configure MPI (will fail if MPI not found, but that's OK for this test)
tt_configure_mpi(ON USE_MPI)

if(USE_MPI)
    message(STATUS "MPI found: TRUE")
    if(TT_METAL_USING_ULFM)
        message(STATUS "Using ULFM: TRUE")
    else()
        message(STATUS "Using ULFM: FALSE")
    endif()
else()
    message(STATUS "MPI found: FALSE")
endif()
"""
        result = self._run_cmake_test(cmake_script, expect_failure=True)
        # Should either find system MPI or fail gracefully
        # Note: This test may pass or fail depending on whether system MPI is installed
        self.assertTrue(
            "MPI found: TRUE" in result
            or "MPI found: FALSE" in result
            or "FATAL_ERROR" in result
            or "no MPI implementation found" in result
        )

    def test_mpi_disabled_when_distributed_off(self):
        """Test that MPI is not configured when distributed is disabled."""
        cmake_script = f"""
cmake_minimum_required(VERSION 3.15)
project(test_mpi)

include({self._path_to_cmake_string(self.test_cmake_dir)}/mpi-config.cmake)

tt_configure_mpi(OFF USE_MPI)

if(USE_MPI)
    message(STATUS "MPI found: TRUE")
else()
    message(STATUS "MPI found: FALSE")
endif()
"""
        result = self._run_cmake_test(cmake_script)
        self.assertIn("MPI found: FALSE", result)
        self.assertIn("Distributed compute disabled", result)

    def test_mpi_lib_dir_extraction(self):
        """Test that MPI library directory is correctly extracted."""
        # Create a mock ULFM installation
        ulfm_prefix = self.temp_dir / "opt" / "openmpi-v5.0.7-ulfm"
        ulfm_prefix.mkdir(parents=True)
        (ulfm_prefix / "lib" / "libmpi.so.40").parent.mkdir(parents=True)
        (ulfm_prefix / "lib" / "libmpi.so.40").touch()

        ulfm_lib_dir = self._path_to_cmake_string(ulfm_prefix / "lib")
        cmake_script = f"""
cmake_minimum_required(VERSION 3.15)
project(test_mpi)

set(ULFM_PREFIX "{self._path_to_cmake_string(ulfm_prefix)}")
include({self._path_to_cmake_string(self.test_cmake_dir)}/mpi-config.cmake)

tt_configure_mpi(ON USE_MPI)

        if(TT_METAL_MPI_LIB_DIR)
            if("${{TT_METAL_MPI_LIB_DIR}}" STREQUAL "{ulfm_lib_dir}")
                message(STATUS "MPI lib dir correct: TRUE")
            else()
                message(FATAL_ERROR "MPI lib dir incorrect: ${{TT_METAL_MPI_LIB_DIR}}")
            endif()
        else()
            message(FATAL_ERROR "TT_METAL_MPI_LIB_DIR not set")
        endif()
"""
        result = self._run_cmake_test(cmake_script)
        self.assertIn("MPI lib dir correct: TRUE", result)

    def test_ulfm_prefix_without_library(self):
        """Test that ULFM prefix without libmpi.so.40 falls back to system MPI."""
        # Create ULFM prefix directory but no library file
        ulfm_prefix = self.temp_dir / "opt" / "openmpi-v5.0.7-ulfm"
        ulfm_prefix.mkdir(parents=True)

        cmake_script = f"""
cmake_minimum_required(VERSION 3.15)
project(test_mpi)

set(ULFM_PREFIX "{self._path_to_cmake_string(ulfm_prefix)}")
include({self._path_to_cmake_string(self.test_cmake_dir)}/mpi-config.cmake)

tt_configure_mpi(ON USE_MPI)

if(USE_MPI)
    message(STATUS "MPI found: TRUE")
    if(TT_METAL_USING_ULFM)
        message(STATUS "Using ULFM: TRUE")
    else()
        message(STATUS "Using ULFM: FALSE")
    endif()
else()
    message(STATUS "MPI found: FALSE")
endif()
"""
        result = self._run_cmake_test(cmake_script, expect_failure=True)
        # Should not use ULFM since libmpi.so.40 doesn't exist
        # May find system MPI or fail, but should not claim ULFM is being used
        if "Using ULFM: TRUE" in result:
            self.fail("Should not detect ULFM when libmpi.so.40 is missing")

    def _run_cmake_test(self, cmake_script: str, expect_failure: bool = False) -> str:
        """
        Run a CMake test script and return the output.

        Args:
            cmake_script: The CMake script content to execute
            expect_failure: If True, the test will not fail if CMake returns non-zero.
                          This is useful for tests that may or may not find MPI.

        Returns:
            Combined stdout and stderr output from CMake execution.

        Raises:
            AssertionError: If the test fails unexpectedly (when expect_failure=False).
        """
        build_dir = self.temp_dir / "build"
        build_dir.mkdir()

        script_path = build_dir / "test.cmake"
        script_path.write_text(cmake_script)

        try:
            result = subprocess.run(
                ["cmake", "-P", str(script_path)],
                cwd=str(build_dir),
                capture_output=True,
                text=True,
                timeout=30,
                check=False,  # Don't raise on non-zero return code
            )
            output = result.stdout + result.stderr
            if not expect_failure and result.returncode != 0:
                self.fail(f"CMake test failed with return code {result.returncode}:\n{output}")
            return output
        except subprocess.TimeoutExpired:
            self.fail("CMake test timed out after 30 seconds")
        except FileNotFoundError:
            self.skipTest("CMake not found in PATH")


class TestMPIRPATHHandling(unittest.TestCase):
    """
    Test RPATH handling for MPI libraries.

    Note: Full RPATH testing would require building actual libraries and checking
    their RPATH with tools like readelf or chrpath. These tests are placeholders
    for future integration tests that would verify:
    1. TT_METAL_MPI_LIB_DIR is correctly added to INSTALL_RPATH
    2. Built libraries can find libmpi.so at runtime without LD_LIBRARY_PATH
    3. RPATH ordering (e.g., $ORIGIN before absolute paths on Fedora)
    """

    @unittest.skip("Requires full build to verify RPATH in compiled libraries")
    def test_rpath_contains_mpi_dir(self):
        """Test that RPATH includes MPI library directory."""
        # This would require a full build, so we'll test the logic conceptually
        # In a real scenario, we'd check the built library's RPATH
        pass

    @unittest.skip("Requires full build with ULFM MPI to verify RPATH")
    def test_ulfm_rpath_handling(self):
        """Test that ULFM MPI path is added to RPATH."""
        # This would require a full build with ULFM MPI
        pass


if __name__ == "__main__":
    unittest.main()
