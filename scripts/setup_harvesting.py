#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
"""
Script to set up firmware with harvesting configuration.
This script:
1. Clones required repositories
2. Sets up tt-flash in a virtual environment
3. Allows user to select firmware version
4. Applies harvesting configuration
5. Flashes the modified firmware
6. Verifies the setup
"""

import subprocess
import sys
import os
import re
import glob
import json
import shutil
import urllib.request
import urllib.error


def run_command(cmd, cwd=None, shell=True, check=True, capture_output=False):
    """Run a shell command and return the result."""
    print(f"\n>>> Running: {cmd}")
    if cwd:
        print(f"    (in directory: {cwd})")

    result = subprocess.run(cmd, shell=shell, cwd=cwd, check=check, capture_output=capture_output, text=True)

    if capture_output:
        return result.stdout.strip()
    return result


def clone_repos():
    """Clone the required repositories."""
    print("\n" + "=" * 80)
    print("STEP 1: Cloning repositories")
    print("=" * 80)

    repos = {
        "tt-flash": "https://github.com/tenstorrent/tt-flash",
        "tt-firmware": "https://github.com/tenstorrent/tt-firmware.git",
        "tt-system-firmware": "https://github.com/tenstorrent/tt-system-firmware.git",
    }

    for name, url in repos.items():
        if os.path.exists(name):
            print(f"\n✓ {name} already exists, skipping clone")
        else:
            print(f"\nCloning {name}...")
            run_command(f"git clone {url}")

    return repos


def setup_tt_flash_venv():
    """Set up tt-flash with a virtual environment."""
    print("\n" + "=" * 80)
    print("STEP 2: Setting up tt-flash virtual environment")
    print("=" * 80)

    tt_flash_dir = "tt-flash"
    venv_path = "tt-flash/venv"

    if not os.path.exists(venv_path):
        print("\nCreating virtual environment...")
        run_command(f"python3 -m venv venv", cwd=tt_flash_dir)
    else:
        print("\n✓ Virtual environment already exists")

    print("\nInstalling tt-flash into the virtual environment...")
    run_command(f"./venv/bin/pip install --upgrade pip", cwd=tt_flash_dir)
    run_command(f"./venv/bin/pip install .", cwd=tt_flash_dir)

    return venv_path


GITHUB_ORG = "tenstorrent"
_USER_AGENT = "tt-setup-harvesting"


def _github_api_json(url):
    """Fetch and parse a GitHub REST API JSON endpoint (public, unauthenticated)."""
    request = urllib.request.Request(
        url,
        headers={"User-Agent": _USER_AGENT, "Accept": "application/vnd.github+json"},
    )
    with urllib.request.urlopen(request) as response:
        return json.load(response)


# Current-epoch firmware bundles are named
# ``fw_pack-<MAJOR>.<MINOR>.<PATCH>[-rcN].fwbundle`` (a 3-component version).
# The legacy v80.x epoch (pre-renumber) instead ships a 4-component
# ``fw_pack-80.x.y.z.fwbundle``, which this pattern deliberately does NOT match
# — that is how the v80.x line is excluded, regardless of how its git tag is
# spelled (``v80.18.1`` in tt-system-firmware, ``v80.18.1.0`` in tt-firmware).
_FW_BUNDLE_RE = re.compile(r"^fw_pack-((\d+)\.\d+\.\d+(?:-rc\d+)?)\.fwbundle$")


def list_released_versions(repo_dir):
    """Return [(tag, version, release_json), ...] of flashable v19+ firmware releases.

    Firmware bundles are published as GitHub *release assets* (they are not all
    committed into the git tree), so enumerating releases is what gives the set
    of flashable versions. A release is kept only when it carries a standard
    ``fw_pack-<major>.<minor>.<patch>[-rcN].fwbundle`` asset whose major is >= 19.
    This:

      * drops the older v18.x line (major < 19),
      * drops the legacy v80.x epoch, whose bundle is a 4-component
        ``fw_pack-80.x.y.z.fwbundle`` that does not match the pattern (true
        whether its tag is 3-part ``v80.18.1`` or 4-part ``v80.18.1.0``),
      * keeps every v19.x release including ``-rc`` ones, plus future v20.x/v21.x
        releases that follow the same naming, and
      * guarantees the download step can find the asset (we matched it here).

    The returned ``version`` is taken from the matched asset (not the tag), so it
    is always exactly what download_release_bundle() will look for.
    """
    url = f"https://api.github.com/repos/{GITHUB_ORG}/{repo_dir}/releases?per_page=100"
    print(f"\nFetching available firmware releases from {repo_dir} (v19 and above)...")
    try:
        releases = _github_api_json(url)
    except (urllib.error.URLError, urllib.error.HTTPError) as exc:
        print(f"\nERROR: Failed to fetch releases for {repo_dir}: {exc}")
        sys.exit(1)

    versions = []
    for release in releases:
        tag = release.get("tag_name", "")
        for asset in release.get("assets", []):
            match = _FW_BUNDLE_RE.match(asset.get("name", ""))
            if match and int(match.group(2)) >= 19:
                versions.append((tag, match.group(1), release))
                break
    return versions


def download_release_bundle(release, version, dest_dir):
    """Download the fw_pack-<version>.fwbundle release asset; return its local path or None."""
    asset_name = f"fw_pack-{version}.fwbundle"
    asset_url = None
    for asset in release.get("assets", []):
        if asset.get("name") == asset_name:
            asset_url = asset.get("browser_download_url")
            break
    if not asset_url:
        return None

    dest_path = os.path.join(dest_dir, asset_name)
    print(f"\nDownloading {asset_name} ...")
    print(f"  from: {asset_url}")
    request = urllib.request.Request(asset_url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request) as response, open(dest_path, "wb") as out:
            shutil.copyfileobj(response, out)
    except (urllib.error.URLError, urllib.error.HTTPError) as exc:
        print(f"\nERROR: Failed to download {asset_name}: {exc}")
        sys.exit(1)
    return dest_path


def find_fwbundle(repo_dir, version):
    """Fallback: locate a firmware bundle committed inside the repo's git tree.

    The archived tt-firmware repo commits ``fw_pack-<version>.fwbundle`` files
    directly (rather than attaching them as release assets), so this is used
    when no matching release asset was found. Returns the path, or None.
    """
    expected = os.path.join(repo_dir, f"fw_pack-{version}.fwbundle")
    if os.path.exists(expected):
        return expected

    candidates = sorted(glob.glob(os.path.join(repo_dir, "**", "*.fwbundle"), recursive=True))
    if len(candidates) == 1:
        print(f"\nNote: expected {expected} not found; using {candidates[0]}")
        return candidates[0]
    if len(candidates) > 1:
        print(f"\nMultiple firmware bundles found in {repo_dir}:")
        for i, path in enumerate(candidates, 1):
            print(f"  {i}. {path}")
        while True:
            choice = input(f"\nSelect firmware bundle (1-{len(candidates)}): ").strip()
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(candidates):
                    return candidates[idx]
            except ValueError:
                pass
            print(f"Please enter a number between 1 and {len(candidates)}")

    return None


def select_firmware_repo():
    """Let user choose which firmware repository to source the bundle from."""
    print("\n" + "=" * 80)
    print("STEP 3: Selecting firmware")
    print("=" * 80)

    print("\n" + "!" * 80)
    print("WARNING: The 'tt-firmware' repository is now ARCHIVED and no longer maintained.")
    print("         Prefer 'tt-system-firmware' for up-to-date firmware.")
    print("!" * 80)

    repos = {
        "1": ("tt-firmware", "tt-firmware  (ARCHIVED - no longer maintained)"),
        "2": ("tt-system-firmware", "tt-system-firmware  (maintained)"),
    }

    print("\nAvailable firmware repositories:")
    for key, (_repo_dir, label) in repos.items():
        print(f"  {key}. {label}")

    while True:
        choice = input("\nSelect firmware repository (1-2): ").strip()
        if choice not in repos:
            print("Invalid choice. Please enter 1 or 2.")
            continue

        repo_dir, _label = repos[choice]
        if repo_dir == "tt-firmware":
            print("\nYou selected an ARCHIVED repository (tt-firmware); it is no longer maintained.")
            confirm = input("Continue with tt-firmware anyway? (yes/no): ").strip().lower()
            if confirm != "yes":
                print("Please choose again.")
                continue

        print(f"\n✓ Selected firmware repository: {repo_dir}")
        return repo_dir


def select_firmware_version(repo_dir):
    """Let user select a v19+ firmware release and obtain its bundle."""
    versions = list_released_versions(repo_dir)

    if not versions:
        print(f"ERROR: No v19+ firmware releases found in {repo_dir} repository")
        sys.exit(1)

    print("\nAvailable firmware versions (v19 and above):")
    for i, (tag, _version, _release) in enumerate(versions, 1):
        print(f"  {i}. {tag}")

    while True:
        try:
            choice = input(f"\nSelect firmware version (1-{len(versions)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(versions):
                selected_tag, version, selected_release = versions[idx]
                break
            else:
                print(f"Please enter a number between 1 and {len(versions)}")
        except (ValueError, KeyboardInterrupt):
            print("\nInvalid input. Please enter a number.")

    # Preferred path: download the bundle from the GitHub release assets
    # (tt-system-firmware ships bundles this way, not committed to the git tree).
    fwbundle_path = download_release_bundle(selected_release, version, dest_dir=os.getcwd())

    # Fallback: the archived tt-firmware commits fw_pack-<version>.fwbundle into
    # the git tree instead of attaching a release asset.
    if fwbundle_path is None:
        print(f"\nNo 'fw_pack-{version}.fwbundle' release asset found; checking the git tree...")
        print(f"Checking out {selected_tag} in {repo_dir}...")
        run_command(f"git checkout {selected_tag}", cwd=repo_dir)
        fwbundle_path = find_fwbundle(repo_dir, version)

    if not fwbundle_path or not os.path.exists(fwbundle_path):
        print(f"\nERROR: Could not obtain a firmware bundle for {selected_tag} from {repo_dir}")
        sys.exit(1)

    print(f"\n✓ Firmware bundle ready: {fwbundle_path}")
    return fwbundle_path, version


def select_chip_type():
    """Let user select chip emulation type."""
    print("\n" + "=" * 80)
    print("STEP 4: Selecting chip emulation type")
    print("=" * 80)

    chip_types = {
        "1": ("Legacy Loudbox", 0, 140),
        "2": ("Galaxy like", 1, 130),
        "3": ("Current Loudbox", 2, 120),
        "4": ("Default", None, None),
    }

    print("\nAvailable chip types:")
    for key, (name, harvesting_cols, expected_chips) in chip_types.items():
        if harvesting_cols is None:
            print(f"  {key}. {name} (flash original firmware)")
        else:
            print(f"  {key}. {name} ({harvesting_cols} harvesting columns, expects {expected_chips} chips)")

    while True:
        choice = input("\nSelect chip type (1-4): ").strip()
        if choice in chip_types:
            name, harvesting_cols, expected_chips = chip_types[choice]
            print(f"\n✓ Selected: {name}")
            return harvesting_cols, expected_chips
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")


def setup_harvesting_tools(venv_path):
    """Install tools needed for harvesting configuration."""
    print("\n" + "=" * 80)
    print("STEP 5: Installing harvesting tools")
    print("=" * 80)

    # Check if protoc is installed
    print("\nChecking for protobuf compiler (protoc)...")
    try:
        subprocess.run(["which", "protoc"], check=True, capture_output=True)
        print("✓ protoc is already installed")
    except subprocess.CalledProcessError:
        print("protoc not found. Installing protobuf-compiler...")
        try:
            run_command("sudo apt-get update && sudo apt-get install -y protobuf-compiler")
            print("✓ protobuf-compiler installed successfully")
        except subprocess.CalledProcessError:
            print("\nERROR: Failed to install protobuf-compiler")
            print("Please install it manually with:")
            print("  sudo apt-get install -y protobuf-compiler")
            sys.exit(1)

    pip_path = f"{venv_path}/bin/pip"

    print("\nInstalling protobuf and tt-flash>=3.6.0...")
    run_command(f"{pip_path} install protobuf 'tt-flash>=3.6.0'")
    run_command(f"{pip_path} install click")

    print("\nInstalling tt-update-tensix-disable-count tool...")
    tool_dir = "tt-system-firmware/scripts/tooling/tt_update_tensix_disable_count"
    run_command(f"{pip_path} install {tool_dir}")


def apply_harvesting(input_fwbundle, harvesting_cols, venv_path):
    """Apply harvesting configuration to firmware."""
    print("\n" + "=" * 80)
    print("STEP 6: Applying harvesting configuration")
    print("=" * 80)

    output_fwbundle = "patched.fwbundle"

    # Use absolute paths to avoid confusion
    cwd = os.getcwd()
    input_absolute = os.path.abspath(input_fwbundle)
    output_absolute = os.path.abspath(output_fwbundle)

    # Verify input file exists
    if not os.path.exists(input_absolute):
        print(f"\nERROR: Input firmware bundle not found at: {input_absolute}")
        sys.exit(1)

    cmd = (
        f". {os.getcwd()}/{venv_path}/bin/activate &&"
        f"tt-system-firmware/scripts/update_tensix_disable_count.py "
        f"--input {input_absolute} "
        f"--output {output_absolute} "
        f"--board P150A-1 --board P150B-1 --board P150C-1 "
        f"--disable-count {harvesting_cols}"
    )

    print(f"\nApplying {harvesting_cols} harvesting column(s)...")
    print(f"Input:  {input_absolute}")
    print(f"Output: {output_absolute}")

    try:
        run_command(cmd, cwd=cwd)
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Command failed with exit code {e.returncode}")
        print("Check the output above for details.")
        sys.exit(1)

    if not os.path.exists(output_absolute):
        print(f"\nERROR: Patched firmware bundle not created at: {output_absolute}")
        print("The command completed but did not produce the output file.")
        sys.exit(1)

    print(f"\n✓ Patched firmware created: {output_fwbundle}")
    return output_fwbundle


def flash_firmware(venv_path, fwbundle_path):
    """Flash the modified firmware."""
    print("\n" + "=" * 80)
    print("STEP 7: Flashing firmware")
    print("=" * 80)

    tt_flash_path = f"{venv_path}/bin/tt-flash"

    cmd = f"{tt_flash_path} {fwbundle_path} --force"

    print("\nFlashing firmware...")
    print("WARNING: This will flash the firmware to your device!")

    confirm = input("Continue? (yes/no): ").strip().lower()
    if confirm != "yes":
        print("\nFlashing cancelled by user.")
        sys.exit(0)

    run_command(cmd)
    print("\n✓ Firmware flashed successfully")


def verify_setup(expected_chips):
    """Verify the firmware setup using pyluwen."""
    print("\n" + "=" * 80)
    print("STEP 8: Verifying setup")
    print("=" * 80)

    print("\nVerifying chip count with pyluwen...")

    try:
        import pyluwen

        chips = pyluwen.detect_chips()
        if not chips:
            print("\nERROR: No chips detected!")
            return False

        num_chips = chips[0].get_telemetry().tensix_enabled_col.bit_count() * 10

        print(f"\nDetected chips: {num_chips}")
        print(f"Expected chips: {expected_chips}")

        if num_chips == expected_chips:
            print("\n✓ SUCCESS! Chip count matches expected value.")
            return True
        else:
            print(f"\n✗ ERROR: Chip count mismatch! The device might not be bin 1")
            print(f"   Expected: {expected_chips}")
            print(f"   Got: {num_chips}")
            return False

    except ImportError:
        print("\nERROR: pyluwen is not installed. Cannot verify setup.")
        print("Please install pyluwen and run verification manually:")
        print("  import pyluwen")
        print("  num_chips = pyluwen.detect_chips()[0].get_telemetry().tensix_enabled_col.bit_count() * 10")
        print(f"  Expected: {expected_chips}")
        return False
    except Exception as e:
        print(f"\nERROR during verification: {e}")
        return False


def main():
    """Main function to orchestrate the entire setup process."""
    print("=" * 80)
    print("FIRMWARE HARVESTING SETUP SCRIPT")
    print("=" * 80)

    try:
        # Step 1: Clone repositories
        clone_repos()

        # Step 2: Setup tt-flash venv
        venv_path = setup_tt_flash_venv()

        # Step 3: Select firmware repository (with archived-repo warning) + version
        repo_dir = select_firmware_repo()
        fwbundle_path, _ = select_firmware_version(repo_dir)

        # Step 4: Select chip type
        harvesting_cols, expected_chips = select_chip_type()

        # Check if Default (no harvesting) was selected
        if harvesting_cols is None:
            print("\n✓ Default mode selected - skipping harvesting steps")
            firmware_to_flash = fwbundle_path
        else:
            # Step 5: Setup harvesting tools
            setup_harvesting_tools(venv_path)

            # Step 6: Apply harvesting
            firmware_to_flash = apply_harvesting(fwbundle_path, harvesting_cols, venv_path)

        # Step 7: Flash firmware
        flash_firmware(venv_path, firmware_to_flash)

        # Step 8: Verify
        if expected_chips is not None:
            success = verify_setup(expected_chips)
        else:
            print("\n" + "=" * 80)
            print("STEP 8: Skipping verification (Default mode)")
            print("=" * 80)
            success = True

        print("\n" + "=" * 80)
        if success:
            print("SETUP COMPLETED SUCCESSFULLY!")
        else:
            print("SETUP COMPLETED WITH WARNINGS - Please verify manually")
        print("=" * 80)

        print("\nTo activate the tt-flash virtual environment in the future, run:")
        print(f"  source {venv_path}/bin/activate")

    except KeyboardInterrupt:
        print("\n\nScript interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
