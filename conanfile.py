# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import subprocess
import re

from conan import ConanFile
from conan.tools.cmake import CMake, CMakeToolchain, cmake_layout


class TTMetaliumConan(ConanFile):
    name = "tt-metalium"
    version = None
    package_type = "library"
    license = "Apache-2.0"
    url = "https://github.com/tenstorrent/tt-metal"
    description = "Tenstorrent Metalium runtime library"
    topics = ("ai", "ml", "runtime", "tenstorrent")

    settings = "os", "arch", "compiler", "build_type"
    options = {
        "shared": [True, False],
        "with_python_bindings": [True, False],
        "build_examples": [True, False],
        "build_tests": [True, False],
        "enable_distributed": [True, False],
        "enable_profiler": [True, False],
        "enable_tt_train": [True, False],
    }
    default_options = {
        "shared": True,
        "with_python_bindings": True,
        "build_examples": False,
        "build_tests": False,
        "enable_distributed": True,
        "enable_profiler": False,
        "enable_tt_train": False,
    }

    def export(self):
        self.exports_sources = [
            "CMakeLists.txt",
            "ttnn/**",
            "tt_metal/**",
            "tt_stl/**",
            "CMakePresets.json",
            "cmake/**",
            "third_party/**",
            ".clang-tidy",
            # exclude build artifacts
            "!**/*.o",
            "!**/*.out",
            "!**/*.bin",
            "!**/*.obj",
            "!**/*.a",
            "!**/*.lib",
            "!**/*.so*",
            "!**/*.dll",
            "!**/*.dylib",
            "!**/*.pdb",
            "!**/*.ninja",
            "!**/*.Dockerfile",
            "!**/__pycache__/**",
        ]

    def set_version(self):
        _version = subprocess.check_output("git describe --abbrev=0 --tags", shell=True).decode().strip()
        m = re.fullmatch(r"[vV]?(\d+\.\d+\.\d+)(?:-rc(\d+))?", _version)
        if not m:
            raise ValueError(f"error: unsupported version format: {m!r}")

        self.version = m.group(1) + ("." + m.group(2) if m.group(2) else "")

    def layout(self):
        # Avoid collisions with any existing 'build' symlink/folder in sources
        cmake_layout(self, build_folder=".conan-build")

    def build_requirements(self):
        self.tool_requires("ninja/[>=1.11.1]")
        self.tool_requires("cmake/[>=3.25]")

    def generate(self):
        tc = CMakeToolchain(self)
        tc.generator = "Ninja"

        tc.variables["BUILD_TT_TRAIN"] = bool(self.options.enable_tt_train)

        tc.variables["TT_UNITY_BUILDS"] = True
        tc.variables["ENABLE_CCACHE"] = False
        tc.variables["ENABLE_BUILD_TIME_TRACE"] = False
        tc.variables["ENABLE_COVERAGE"] = False
        tc.variables["CMAKE_EXPORT_COMPILE_COMMANDS"] = False
        tc.variables["ENABLE_TRACY"] = self.options.enable_profiler
        tc.variables["TT_ENABLE_LIGHT_METAL_TRACE"] = True

        #########################################################
        tc.variables["TT_INSTALL"] = True
        tc.variables["WITH_PYTHON_BINDINGS"] = bool(self.options.with_python_bindings)
        tc.variables["ENABLE_DISTRIBUTED"] = bool(self.options.enable_distributed)
        tc.variables["ENABLE_FAKE_KERNELS_TARGET"] = False

        pkg_folder = getattr(self, "package_folder", None)
        if pkg_folder:
            tc.variables["CMAKE_INSTALL_PREFIX"] = str(Path(pkg_folder))

        # It affects only third party libraries (boost/googletest/nanomsg/libuv)
        tc.variables["BUILD_SHARED_LIBS"] = True
        tc.variables["ENABLE_TTNN_SHARED_SUBLIBS"] = False

        # Tests
        tc.variables["TT_METAL_BUILD_TESTS"] = bool(self.options.build_tests)
        tc.variables["TTNN_BUILD_TESTS"] = bool(self.options.build_tests)
        tc.variables["BUILD_PROGRAMMING_EXAMPLES"] = bool(self.options.build_examples)

        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure(variables={"VERSION_NUMERIC": self.version})
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        # If profiler(Tracy) is enabled, we need to propagate additional compiler flags to the user of the library
        if self.options.enable_profiler:
            self.cpp_info.cxxflags = ["-fno-omit-frame-pointer"]
            self.cpp_info.link_options = ["-rdynamic"]

        self.cpp_info.libs = ["tt_stl", "ttnn", "tt_metal"]

        self.cpp_info.components["tt_stl"].libs = ["tt_stl"]
        self.cpp_info.components["tt_stl"].defines = ["SPDLOG_FMT_EXTERNAL", "FMT_HEADER_ONLY"]

        self.cpp_info.components["tt-metalium"].libs = ["tt_metal"]
        self.cpp_info.components["tt-metalium"].requires = ["tt_stl"]

        self.cpp_info.components["ttnn"].libs = ["ttnn"]
        self.cpp_info.components["ttnn"].requires = ["tt-metalium"]

        self.runenv_info.define("TT_METAL_HOME", str(self.package_folder) + "/bin/tt-metalium/")
