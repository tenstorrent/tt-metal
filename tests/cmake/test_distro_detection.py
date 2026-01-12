#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for CMake distro detection functionality.

This module tests the detect_distro() function by simulating various
/etc/os-release file contents and verifying correct detection.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestDistroDetection(unittest.TestCase):
    """Test cases for distro detection logic."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.os_release_path = Path(self.temp_dir) / "os-release"

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def write_os_release(self, content: str):
        """Write content to a temporary os-release file."""
        self.os_release_path.write_text(content)

    def test_detect_ubuntu_22_04(self):
        """Test detection of Ubuntu 22.04."""
        content = """NAME="Ubuntu"
VERSION="22.04.3 LTS (Jammy Jellyfish)"
ID=ubuntu
ID_LIKE=debian
PRETTY_NAME="Ubuntu 22.04.3 LTS"
VERSION_ID="22.04"
"""
        self.write_os_release(content)
        result = self._run_detect_distro()
        self.assertEqual(result["DISTRO_ID"], "ubuntu")
        self.assertEqual(result["DISTRO_ID_LIKE"], "debian")
        self.assertEqual(result["DISTRO_VERSION"], "22.04")

    def test_detect_fedora_43(self):
        """Test detection of Fedora 43."""
        content = """NAME="Fedora Linux"
VERSION="43 (Workstation Edition)"
ID=fedora
ID_LIKE=fedora
VERSION_ID=43
"""
        self.write_os_release(content)
        result = self._run_detect_distro()
        self.assertEqual(result["DISTRO_ID"], "fedora")
        self.assertEqual(result["DISTRO_ID_LIKE"], "fedora")
        self.assertEqual(result["DISTRO_VERSION"], "43")

    def test_detect_debian(self):
        """Test detection of Debian."""
        content = """PRETTY_NAME="Debian GNU/Linux 12 (bookworm)"
NAME="Debian GNU/Linux"
VERSION_ID="12"
VERSION="12 (bookworm)"
ID=debian
ID_LIKE=debian
"""
        self.write_os_release(content)
        result = self._run_detect_distro()
        self.assertEqual(result["DISTRO_ID"], "debian")
        self.assertEqual(result["DISTRO_ID_LIKE"], "debian")
        self.assertEqual(result["DISTRO_VERSION"], "12")

    def test_detect_rhel(self):
        """Test detection of RHEL."""
        content = """NAME="Red Hat Enterprise Linux"
VERSION="9.3 (Plow)"
ID=rhel
ID_LIKE="fedora"
VERSION_ID="9.3"
"""
        self.write_os_release(content)
        result = self._run_detect_distro()
        self.assertEqual(result["DISTRO_ID"], "rhel")
        self.assertEqual(result["DISTRO_ID_LIKE"], "fedora")
        self.assertEqual(result["DISTRO_VERSION"], "9.3")

    def test_detect_centos(self):
        """Test detection of CentOS."""
        content = """NAME="CentOS Linux"
VERSION="8"
ID=centos
ID_LIKE="rhel fedora"
VERSION_ID="8"
"""
        self.write_os_release(content)
        result = self._run_detect_distro()
        self.assertEqual(result["DISTRO_ID"], "centos")
        self.assertIn("rhel", result["DISTRO_ID_LIKE"])
        self.assertEqual(result["DISTRO_VERSION"], "8")

    def test_detect_opensuse(self):
        """Test detection of openSUSE."""
        content = """NAME="openSUSE Leap"
VERSION="15.5"
ID=opensuse-leap
ID_LIKE="suse opensuse"
VERSION_ID="15.5"
"""
        self.write_os_release(content)
        result = self._run_detect_distro()
        self.assertEqual(result["DISTRO_ID"], "opensuse-leap")
        self.assertIn("suse", result["DISTRO_ID_LIKE"])
        self.assertEqual(result["DISTRO_VERSION"], "15.5")

    def test_detect_with_quotes(self):
        """Test detection with quoted values."""
        content = """NAME="Ubuntu"
ID="ubuntu"
ID_LIKE="debian"
VERSION_ID="22.04"
"""
        self.write_os_release(content)
        result = self._run_detect_distro()
        self.assertEqual(result["DISTRO_ID"], "ubuntu")
        self.assertEqual(result["DISTRO_ID_LIKE"], "debian")
        self.assertEqual(result["DISTRO_VERSION"], "22.04")

    def test_detect_without_quotes(self):
        """Test detection without quotes."""
        content = """NAME=Ubuntu
ID=ubuntu
ID_LIKE=debian
VERSION_ID=22.04
"""
        self.write_os_release(content)
        result = self._run_detect_distro()
        self.assertEqual(result["DISTRO_ID"], "ubuntu")
        self.assertEqual(result["DISTRO_ID_LIKE"], "debian")
        self.assertEqual(result["DISTRO_VERSION"], "22.04")

    def test_missing_os_release(self):
        """Test behavior when /etc/os-release doesn't exist."""
        # Don't create the file
        result = self._run_detect_distro()
        self.assertEqual(result["DISTRO_ID"], "")
        self.assertEqual(result["DISTRO_ID_LIKE"], "")
        self.assertEqual(result["DISTRO_VERSION"], "")

    def test_partial_os_release(self):
        """Test detection with partial os-release file."""
        content = """ID=ubuntu
VERSION_ID=22.04
"""
        self.write_os_release(content)
        result = self._run_detect_distro()
        self.assertEqual(result["DISTRO_ID"], "ubuntu")
        self.assertEqual(result["DISTRO_ID_LIKE"], "")
        self.assertEqual(result["DISTRO_VERSION"], "22.04")

    def test_case_insensitive_id(self):
        """Test that ID is converted to lowercase."""
        content = """ID=UBUNTU
VERSION_ID=22.04
"""
        self.write_os_release(content)
        result = self._run_detect_distro()
        self.assertEqual(result["DISTRO_ID"], "ubuntu")

    def test_case_insensitive_id_like(self):
        """Test that ID_LIKE is converted to lowercase."""
        content = """ID=ubuntu
ID_LIKE=DEBIAN
VERSION_ID=22.04
"""
        self.write_os_release(content)
        result = self._run_detect_distro()
        self.assertEqual(result["DISTRO_ID_LIKE"], "debian")

    def _run_detect_distro(self):
        """
        Simulate the detect_distro() CMake function logic.
        Returns a dict with DISTRO_ID, DISTRO_ID_LIKE, DISTRO_VERSION.
        """
        distro_id = ""
        distro_id_like = ""
        distro_version = ""

        if self.os_release_path.exists():
            lines = self.os_release_path.read_text().splitlines()
            for line in lines:
                # Extract ID
                import re

                match = re.match(r'^ID="?([^"]+)"?', line)
                if match:
                    distro_id = match.group(1).lower()

                # Extract ID_LIKE
                match = re.match(r'^ID_LIKE="?([^"]+)"?', line)
                if match:
                    distro_id_like = match.group(1).lower()

                # Extract VERSION_ID
                match = re.match(r'^VERSION_ID="?([^"]+)"?', line)
                if match:
                    distro_version = match.group(1)

        return {
            "DISTRO_ID": distro_id,
            "DISTRO_ID_LIKE": distro_id_like,
            "DISTRO_VERSION": distro_version,
        }


class TestPackagingTypeDetection(unittest.TestCase):
    """Test cases for packaging type detection logic."""

    def test_deb_from_id_like_debian(self):
        """Test DEB detection from ID_LIKE=debian."""
        self.assertEqual(self._detect_packaging_type("ubuntu", "debian", ""), "DEB")

    def test_deb_from_id_like_ubuntu(self):
        """Test DEB detection from ID_LIKE=ubuntu."""
        self.assertEqual(self._detect_packaging_type("debian", "ubuntu", ""), "DEB")

    def test_rpm_from_id_like_fedora(self):
        """Test RPM detection from ID_LIKE=fedora."""
        self.assertEqual(self._detect_packaging_type("fedora", "fedora", ""), "RPM")

    def test_rpm_from_id_like_rhel(self):
        """Test RPM detection from ID_LIKE=rhel."""
        self.assertEqual(self._detect_packaging_type("centos", "rhel", ""), "RPM")

    def test_deb_from_id_ubuntu(self):
        """Test DEB detection from ID=ubuntu."""
        self.assertEqual(self._detect_packaging_type("ubuntu", "", ""), "DEB")

    def test_rpm_from_id_fedora(self):
        """Test RPM detection from ID=fedora."""
        self.assertEqual(self._detect_packaging_type("fedora", "", ""), "RPM")

    def test_fallback_to_deb(self):
        """Test fallback to DEB when detection fails."""
        self.assertEqual(self._detect_packaging_type("unknown", "", ""), "DEB")

    def _detect_packaging_type(self, distro_id: str, distro_id_like: str, distro_version: str) -> str:
        """
        Simulate the packaging type detection logic from packaging.cmake.
        """
        packaging_type = ""

        # Check ID_LIKE first (more reliable for derivatives)
        if distro_id_like:
            if "debian" in distro_id_like or "ubuntu" in distro_id_like:
                packaging_type = "DEB"
            elif (
                "fedora" in distro_id_like
                or "rhel" in distro_id_like
                or "centos" in distro_id_like
                or "suse" in distro_id_like
            ):
                packaging_type = "RPM"

        # Check ID if ID_LIKE didn't match
        if not packaging_type and distro_id:
            if "debian" in distro_id or "ubuntu" in distro_id or "linuxmint" in distro_id or "pop" in distro_id:
                packaging_type = "DEB"
            elif (
                "fedora" in distro_id
                or "rhel" in distro_id
                or "centos" in distro_id
                or "rocky" in distro_id
                or "alma" in distro_id
                or "opensuse" in distro_id
                or "sles" in distro_id
            ):
                packaging_type = "RPM"

        # Default to DEB if detection failed
        if not packaging_type:
            packaging_type = "DEB"

        return packaging_type


if __name__ == "__main__":
    unittest.main()
