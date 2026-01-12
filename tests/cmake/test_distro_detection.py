#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for CMake distro detection functionality.

This module tests the detect_distro() function by simulating various
/etc/os-release file contents and verifying correct detection.
"""

import re
import shutil
import tempfile
import unittest
from pathlib import Path


class TestDistroDetection(unittest.TestCase):
    """Test cases for distro detection logic."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.os_release_path = self.temp_dir / "os-release"

    def tearDown(self):
        """Clean up test fixtures."""
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

    def test_detect_ubuntu_24_04_real_world(self):
        """Test detection of Ubuntu 24.04 using real-world os-release format."""
        # Based on actual Ubuntu 24.04.3 LTS /etc/os-release
        content = """PRETTY_NAME="Ubuntu 24.04.3 LTS"
NAME="Ubuntu"
VERSION_ID="24.04"
VERSION="24.04.3 LTS (Noble Numbat)"
VERSION_CODENAME=noble
ID=ubuntu
ID_LIKE=debian
HOME_URL="https://www.ubuntu.com/"
SUPPORT_URL="https://help.ubuntu.com/"
BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"
PRIVACY_POLICY_URL="https://www.ubuntu.com/legal/terms-and-policies/privacy-policy"
UBUNTU_CODENAME=noble
LOGO=ubuntu-logo
"""
        self.write_os_release(content)
        result = self._run_detect_distro()
        self.assertEqual(result["DISTRO_ID"], "ubuntu")
        self.assertEqual(result["DISTRO_ID_LIKE"], "debian")
        self.assertEqual(result["DISTRO_VERSION"], "24.04")

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

    def test_detect_fedora_42_real_world(self):
        """Test detection of Fedora 43 using real-world os-release format."""
        # Based on actual Fedora 42 Container Image /etc/os-release
        # Note: Real Fedora often doesn't have ID_LIKE field
        content = """NAME="Fedora Linux"
VERSION="43 (Container Image)"
RELEASE_TYPE=stable
ID=fedora
VERSION_ID=43
VERSION_CODENAME=""
PLATFORM_ID="platform:f43"
PRETTY_NAME="Fedora Linux 43 (Container Image)"
ANSI_COLOR="0;38;2;60;110;180"
LOGO=fedora-logo-icon
CPE_NAME="cpe:/o:fedoraproject:fedora:43"
DEFAULT_HOSTNAME="fedora"
HOME_URL="https://fedoraproject.org/"
DOCUMENTATION_URL="https://docs.fedoraproject.org/en-US/fedora/f42/"
SUPPORT_URL="https://ask.fedoraproject.org/"
BUG_REPORT_URL="https://bugzilla.redhat.com/"
REDHAT_BUGZILLA_PRODUCT="Fedora"
REDHAT_BUGZILLA_PRODUCT_VERSION=43
REDHAT_SUPPORT_PRODUCT="Fedora"
REDHAT_SUPPORT_PRODUCT_VERSION=43
SUPPORT_END=2026-12-02
VARIANT="Container Image"
VARIANT_ID=container
"""
        self.write_os_release(content)
        result = self._run_detect_distro()
        self.assertEqual(result["DISTRO_ID"], "fedora")
        # Real Fedora 42 doesn't have ID_LIKE field, so it should be empty
        self.assertEqual(result["DISTRO_ID_LIKE"], "")
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
        """Test detection of CentOS with multi-value ID_LIKE."""
        content = """NAME="CentOS Linux"
VERSION="8"
ID=centos
ID_LIKE="rhel fedora"
VERSION_ID="8"
"""
        self.write_os_release(content)
        result = self._run_detect_distro()
        self.assertEqual(result["DISTRO_ID"], "centos")
        # ID_LIKE should contain the full multi-value string
        self.assertEqual(result["DISTRO_ID_LIKE"], "rhel fedora")
        self.assertIn("rhel", result["DISTRO_ID_LIKE"])
        self.assertIn("fedora", result["DISTRO_ID_LIKE"])
        self.assertEqual(result["DISTRO_VERSION"], "8")

    def test_detect_opensuse(self):
        """Test detection of openSUSE with multi-value ID_LIKE."""
        content = """NAME="openSUSE Leap"
VERSION="15.5"
ID=opensuse-leap
ID_LIKE="suse opensuse"
VERSION_ID="15.5"
"""
        self.write_os_release(content)
        result = self._run_detect_distro()
        self.assertEqual(result["DISTRO_ID"], "opensuse-leap")
        # ID_LIKE should contain the full multi-value string
        self.assertEqual(result["DISTRO_ID_LIKE"], "suse opensuse")
        self.assertIn("suse", result["DISTRO_ID_LIKE"])
        self.assertIn("opensuse", result["DISTRO_ID_LIKE"])
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

    def test_multi_value_id_like(self):
        """Test that ID_LIKE with multiple values is preserved correctly."""
        content = """ID=centos
ID_LIKE="rhel fedora"
VERSION_ID=8
"""
        self.write_os_release(content)
        result = self._run_detect_distro()
        # ID_LIKE should preserve the full multi-value string
        self.assertEqual(result["DISTRO_ID_LIKE"], "rhel fedora")
        self.assertIn("rhel", result["DISTRO_ID_LIKE"])
        self.assertIn("fedora", result["DISTRO_ID_LIKE"])

    def test_multi_value_id_like_unquoted(self):
        """Test that unquoted multi-value ID_LIKE is handled correctly."""
        content = """ID=centos
ID_LIKE=rhel fedora
VERSION_ID=8
"""
        self.write_os_release(content)
        result = self._run_detect_distro()
        # Unquoted values should still capture the full line content
        self.assertEqual(result["DISTRO_ID_LIKE"], "rhel fedora")

    def test_empty_values(self):
        """Test handling of empty values in os-release."""
        content = """ID=
ID_LIKE=
VERSION_ID=
"""
        self.write_os_release(content)
        result = self._run_detect_distro()
        # Empty values should result in empty strings
        self.assertEqual(result["DISTRO_ID"], "")
        self.assertEqual(result["DISTRO_ID_LIKE"], "")
        self.assertEqual(result["DISTRO_VERSION"], "")

    def test_whitespace_handling(self):
        """Test that leading/trailing whitespace is handled correctly."""
        content = """ID=  ubuntu
ID_LIKE=  debian
VERSION_ID=  22.04
"""
        self.write_os_release(content)
        result = self._run_detect_distro()
        # The regex should capture whitespace, but CMake's string(TOLOWER) may trim
        # For now, we test what the regex actually captures
        self.assertIn("ubuntu", result["DISTRO_ID"])
        self.assertIn("debian", result["DISTRO_ID_LIKE"])

    def test_special_characters_in_version(self):
        """Test handling of special characters in VERSION_ID."""
        content = """ID=ubuntu
VERSION_ID="22.04.1"
"""
        self.write_os_release(content)
        result = self._run_detect_distro()
        # Version with dots should be preserved
        self.assertEqual(result["DISTRO_VERSION"], "22.04.1")

    def test_comments_in_os_release(self):
        """Test that lines starting with # are ignored (os-release spec)."""
        content = """# This is a comment
ID=ubuntu
# Another comment
VERSION_ID=22.04
"""
        self.write_os_release(content)
        result = self._run_detect_distro()
        # Comments should not interfere with parsing
        self.assertEqual(result["DISTRO_ID"], "ubuntu")
        self.assertEqual(result["DISTRO_VERSION"], "22.04")

    def test_fedora_without_id_like(self):
        """Test Fedora detection when ID_LIKE field is missing (common in real systems)."""
        # Many real Fedora systems don't include ID_LIKE
        content = """NAME="Fedora Linux"
VERSION="42 (Container Image)"
ID=fedora
VERSION_ID=42
"""
        self.write_os_release(content)
        result = self._run_detect_distro()
        self.assertEqual(result["DISTRO_ID"], "fedora")
        self.assertEqual(result["DISTRO_ID_LIKE"], "")
        self.assertEqual(result["DISTRO_VERSION"], "42")

    def test_ubuntu_with_quoted_version_id(self):
        """Test Ubuntu with quoted VERSION_ID (as seen in real systems)."""
        # Real Ubuntu 24.04 has VERSION_ID="24.04" with quotes
        content = """PRETTY_NAME="Ubuntu 24.04.3 LTS"
NAME="Ubuntu"
VERSION_ID="24.04"
ID=ubuntu
ID_LIKE=debian
"""
        self.write_os_release(content)
        result = self._run_detect_distro()
        self.assertEqual(result["DISTRO_ID"], "ubuntu")
        self.assertEqual(result["DISTRO_ID_LIKE"], "debian")
        self.assertEqual(result["DISTRO_VERSION"], "24.04")

    def test_fedora_with_unquoted_version_id(self):
        """Test Fedora with unquoted VERSION_ID (as seen in real systems)."""
        # Real Fedora 42 has VERSION_ID=42 without quotes
        content = """NAME="Fedora Linux"
VERSION="42 (Container Image)"
ID=fedora
VERSION_ID=42
"""
        self.write_os_release(content)
        result = self._run_detect_distro()
        self.assertEqual(result["DISTRO_ID"], "fedora")
        self.assertEqual(result["DISTRO_VERSION"], "42")

    def _run_detect_distro(self):
        """
        Simulate the detect_distro() CMake function logic.

        This replicates the behavior of cmake/detect-distro.cmake:
        - Reads os-release file line by line
        - Extracts ID, ID_LIKE, and VERSION_ID using regex matching
        - Converts ID and ID_LIKE to lowercase
        - Handles quoted and unquoted values

        Note: The CMake regex `^ID_LIKE="?([^"]+)"?` will match the entire
        value including spaces (e.g., "rhel fedora"), which is correct for
        multi-value ID_LIKE fields.

        Returns:
            dict with keys: DISTRO_ID, DISTRO_ID_LIKE, DISTRO_VERSION
        """
        distro_id = ""
        distro_id_like = ""
        distro_version = ""

        if self.os_release_path.exists():
            lines = self.os_release_path.read_text().splitlines()
            for line in lines:
                # Extract ID - matches ID=value or ID="value"
                # The regex [^"]+ matches everything except quotes, which works
                # for both quoted and unquoted single values
                match = re.match(r'^ID="?([^"]+)"?', line)
                if match:
                    distro_id = match.group(1).lower()

                # Extract ID_LIKE - can contain multiple space-separated values
                # The regex [^"]+ will capture everything until the closing quote
                # or end of line, preserving spaces in multi-value fields like
                # ID_LIKE="rhel fedora"
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
