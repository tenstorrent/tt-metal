#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
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
        "tt-system-firmware": "git@github.com:tenstorrent/tt-system-firmware.git",
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

    print("\nInstalling tt-flash in editable mode...")
    run_command(f"./venv/bin/pip install --upgrade pip", cwd=tt_flash_dir)
    run_command(f"./venv/bin/pip install .", cwd=tt_flash_dir)

    return venv_path


def select_firmware_version():
    """Let user select firmware version from available tags."""
    print("\n" + "=" * 80)
    print("STEP 3: Selecting firmware version")
    print("=" * 80)

    print("\nFetching available firmware versions...")
    tags_output = run_command("git tag -l 'v19.*'", cwd="tt-firmware", capture_output=True)

    tags = [tag.strip() for tag in tags_output.split("\n") if tag.strip()]

    if not tags:
        print("ERROR: No v19.x.y tags found in tt-firmware repository")
        sys.exit(1)

    print("\nAvailable firmware versions:")
    for i, tag in enumerate(tags, 1):
        print(f"  {i}. {tag}")

    while True:
        try:
            choice = input(f"\nSelect firmware version (1-{len(tags)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(tags):
                selected_tag = tags[idx]
                break
            else:
                print(f"Please enter a number between 1 and {len(tags)}")
        except (ValueError, KeyboardInterrupt):
            print("\nInvalid input. Please enter a number.")

    print(f"\nChecking out {selected_tag}...")
    run_command(f"git checkout {selected_tag}", cwd="tt-firmware")

    # Extract version number (remove 'v' prefix)
    version = selected_tag[1:] if selected_tag.startswith("v") else selected_tag
    fwbundle_path = f"tt-firmware/fw_pack-{version}.fwbundle"

    if not os.path.exists(fwbundle_path):
        print(f"\nERROR: Expected firmware bundle not found at: {fwbundle_path}")
        sys.exit(1)

    print(f"\n✓ Firmware bundle found: {fwbundle_path}")
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

    print("\nInstalling tt-update-tensix-disable-count tool...")
    tool_dir = "tt-system-firmware/scripts/tooling/tt_update_tensix_disable_count"
    run_command(f"{pip_path} install {tool_dir}")


def apply_harvesting(input_fwbundle, harvesting_cols):
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
        f"tt-update-tensix-disable-count "
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

        # Step 3: Select firmware version
        fwbundle_path, _ = select_firmware_version()

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
            firmware_to_flash = apply_harvesting(fwbundle_path, harvesting_cols)

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
