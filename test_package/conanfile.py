# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from conan import ConanFile
from conan.tools.cmake import cmake_layout, CMake
from conan.tools.env import VirtualRunEnv
from conan.tools.build import can_run

import os


class TestPackageConan(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeDeps", "VirtualBuildEnv"
    test_type = "explicit"

    def requirements(self):
        self.requires(self.tested_reference_str)

    def build_requirements(self):
        self.tool_requires("ninja/[>=1.11.1]")
        self.tool_requires("cmake/[>=3.25]")

    def layout(self):
        cmake_layout(self)

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def generate(self):
        from conan.tools.cmake import CMakeToolchain

        tc = CMakeToolchain(self)
        tc.generator = "Ninja"
        tc.generate()

        runenv = VirtualRunEnv(self)
        runenv.generate()

    def test(self):
        if can_run(self):
            bin_path = os.path.join(self.cpp.build.bindirs[0], "test_package")
            self.run(bin_path, env="conanrun")
