# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import subprocess
import re

from conan import ConanFile
from conan.tools.cmake import CMake, CMakeToolchain, cmake_layout, CMakeDeps
from conan.tools.files import update_conandata


class TTNNConan(ConanFile):
    name = "tt-nn"
    package_type = "library"
    license = "Apache-2.0"
    url = "https://github.com/tenstorrent/tt-metal"
    description = "Tenstorrent Neural Network library"
    topics = ("ai", "ml", "runtime", "tenstorrent")

    settings = "os", "arch", "compiler", "build_type"
    options = {
        "shared": [True, False],
        "build_examples": [True, False],
        "build_tests": [True, False],
        "enable_distributed": [True, False],
        "enable_profiler": [True, False],
    }
    default_options = {
        "shared": True,
        "build_examples": False,
        "build_tests": False,
        "enable_distributed": True,
        "enable_profiler": False,
    }

    exports_sources = [
        "CMakeLists.txt",
        "ttnn/**",
        "tt_metal/**",
        "tt_stl/**",
        "CMakePresets.json",
        "cmake/**",
        "third_party/**",
        "tools/**",
        ".clang-tidy",
        # exclude build artifacts
        "!**/*.o",
        "!**/*.out",
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

    def export(self):
        self.git_commit = subprocess.check_output("git rev-parse --short=10 HEAD", shell=True).decode().strip()
        self.output.info("Captured git commit: {}".format(self.git_commit))
        update_conandata(self, {"scm": {"commit": self.git_commit}})

    def set_version(self):
        _version = subprocess.check_output("git describe --abbrev=0 --tags", shell=True).decode().strip()
        m = re.fullmatch(r"[vV]?(\d+\.\d+\.\d+).*", _version)
        if not m:
            raise ValueError(f"error: unsupported version format: {m!r}")

        self.version = m.group(1)

    def layout(self):
        # Avoid collisions with any existing 'build' symlink/folder in sources
        cmake_layout(self, build_folder=".conan-build")

    def build_requirements(self):
        self.tool_requires("ninja/[>=1.11.1]")
        self.tool_requires("cmake/[>=3.25]")

    def generate(self):
        deps = CMakeDeps(self)
        deps.generate()

        tc = CMakeToolchain(self)
        tc.generator = "Ninja"

        tc.variables["BUILD_TT_TRAIN"] = False

        tc.variables["TT_UNITY_BUILDS"] = True
        tc.variables["ENABLE_CCACHE"] = False
        tc.variables["ENABLE_BUILD_TIME_TRACE"] = False
        tc.variables["ENABLE_COVERAGE"] = False
        tc.variables["CMAKE_EXPORT_COMPILE_COMMANDS"] = False
        tc.variables["ENABLE_TRACY"] = self.options.enable_profiler
        tc.variables["TT_ENABLE_LIGHT_METAL_TRACE"] = True

        #########################################################
        tc.variables["TT_INSTALL"] = True
        tc.variables["WITH_PYTHON_BINDINGS"] = False
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
        # Get version hash from conan_data if available, otherwise get it from git directly
        if self.conan_data and "scm" in self.conan_data and "commit" in self.conan_data["scm"]:
            version_hash = self.conan_data["scm"]["commit"]
        else:
            # Fallback: get git commit directly
            version_hash = subprocess.check_output("git rev-parse --short=10 HEAD", shell=True).decode().strip()
        cmake.configure(variables={"VERSION_NUMERIC": self.version, "VERSION_HASH": version_hash})
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()

        # Core components (always installed)
        components = [
            "metalium-runtime",  # tt_stl, tt_metal libraries, jitapi, hw files, fabric, etc.
            "metalium-dev",  # Headers, CMake config files (tt-metalium-config.cmake), third-party headers
            "ttnn-runtime",  # tt-nn library, kernels, operation libraries
            "ttnn-dev",  # tt-nn headers, CMake config files (tt-nn-config.cmake), operation headers
            "jit-build",  # JIT runtime files (sfpi)
        ]

        # Optional components based on build options
        if self.options.build_examples:
            components.extend(
                [
                    "metalium-examples",  # Programming examples
                    "ttnn-examples",  # TTNN examples
                ]
            )

        if self.options.build_tests:
            components.extend(
                [
                    "metalium-validation",  # Validation tools
                    "ttnn-validation",  # TTNN validation tools
                ]
            )

        for component in components:
            cmake.install(component=component)

    def package_info(self):
        # # If profiler(Tracy) is enabled, we need to propagate additional compiler flags to the user of the library
        if self.options.enable_profiler:
            self.cpp_info.cxxflags = ["-fno-omit-frame-pointer"]
            self.cpp_info.link_options = ["-rdynamic"]

        # Check if we're in editable mode (package_folder points to source folder)
        is_editable = not (Path(self.package_folder) / "bin").exists()

        if is_editable:
            # In editable mode, use a local install directory for proper header structure
            build_folder = Path(self.package_folder) / ".conan-build" / str(self.settings.build_type)
            local_install = Path(self.package_folder) / "install"

            # Use the local install structure (must be created manually with cmake install)
            if not (local_install / "include").exists():
                self.output.warning(
                    f"Local install not found at {local_install}/include. Please run: cd {build_folder} && cmake --install . --prefix {local_install}"
                )

            # Use the installed structure (same as package mode but from local install)
            common_includedirs = [
                str(local_install / "include" / "metalium-thirdparty"),
                str(local_install / "include"),
            ]
            common_defines = ["SPDLOG_FMT_EXTERNAL", "FMT_HEADER_ONLY"]
            # Add common lib directory for tracy, device, etc.
            common_libdir = str(build_folder / "lib")

            self.cpp_info.components["tt_stl"].libs = ["tt_stl"]
            self.cpp_info.components["tt_stl"].libdirs = [str(build_folder / "tt_stl"), common_libdir]
            self.cpp_info.components["tt_stl"].includedirs = common_includedirs
            self.cpp_info.components["tt_stl"].defines = common_defines

            self.cpp_info.components["tt_metal"].libs = ["tt_metal"]
            self.cpp_info.components["tt_metal"].libdirs = [str(build_folder / "tt_metal"), common_libdir]
            self.cpp_info.components["tt_metal"].includedirs = common_includedirs
            self.cpp_info.components["tt_metal"].defines = common_defines
            self.cpp_info.components["tt_metal"].requires = ["tt_stl"]

            self.cpp_info.components["tt-nn"].libs = ["tt-nn"]
            self.cpp_info.components["tt-nn"].libdirs = [str(build_folder / "ttnn"), common_libdir]
            self.cpp_info.components["tt-nn"].includedirs = common_includedirs
            self.cpp_info.components["tt-nn"].defines = common_defines
            self.cpp_info.components["tt-nn"].requires = ["tt_metal", "tt_stl"]

            self.cpp_info.bindirs = [str(build_folder)]
            tt_metal_home = str(local_install / "bin" / "tt-metalium")
        else:
            # In package mode, use installed structure
            self.output.info(f"In package mode, use installed structure: {self.package_folder}")
            self.cpp_info.includedirs = ["include/metalium-thirdparty", "include"]
            self.cpp_info.libs = ["tt_stl", "tt-nn", "tt_metal"]
            tt_metal_home = str(self.package_folder) + "/bin/tt-metalium/"

        self.cpp_info.defines = ["SPDLOG_FMT_EXTERNAL", "FMT_HEADER_ONLY"]
        self.runenv_info.define("TT_METAL_RUNTIME_ROOT", tt_metal_home)
