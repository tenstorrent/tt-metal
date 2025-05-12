# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import pytest
import requests
import subprocess
import sys

from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.log_utils import _format_log

from ttexalens.tt_exalens_lib import arc_msg


def set_chip_architecture():
    def _identify_chip_architecture(output):
        if "Blackhole" in output:
            return ChipArchitecture.BLACKHOLE
        elif "Wormhole" in output:
            return ChipArchitecture.WORMHOLE
        return None

    chip_arch = get_chip_architecture()
    if chip_arch:
        print(f"CHIP_ARCH is already set to {chip_arch}")
        return chip_arch
    try:
        result = subprocess.run(
            ["tt-smi", "-ls"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        print("Error: tt-smi command not found.", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error: tt-smi failed with error: {e.stderr}", file=sys.stderr)
        sys.exit(1)

    architecture = _identify_chip_architecture(result.stdout)
    if not architecture:
        print(
            "Error: Unable to detect architecture from tt-smi output.", file=sys.stderr
        )
        sys.exit(1)
    os.environ["CHIP_ARCH"] = architecture.value
    return architecture


@pytest.fixture(scope="session", autouse=True)
def download_headers():
    HEADER_DIR = "../hw_specific/inc"
    STAMP_FILE = os.path.join(HEADER_DIR, ".headers_downloaded")
    if os.path.exists(STAMP_FILE):
        print("Headers already downloaded. Skipping download.")
        return

    CHIP_ARCH = set_chip_architecture()
    BASE_URL = f"https://raw.githubusercontent.com/tenstorrent/tt-metal/refs/heads/main/tt_metal/hw/inc/{CHIP_ARCH.value}"
    BASE_URL_WORMHOLE_SPECIFIC = f"https://raw.githubusercontent.com/tenstorrent/tt-metal/refs/heads/main/tt_metal/hw/inc/{CHIP_ARCH.value}/wormhole_b0_defines"
    HEADERS = [
        "cfg_defines.h",
        "dev_mem_map.h",
        "tensix.h",
        "tensix_types.h",
    ]

    # Create the header directory if it doesn't exist
    os.makedirs(HEADER_DIR, exist_ok=True)

    # Determine the specific URL based on CHIP_ARCH
    if CHIP_ARCH == ChipArchitecture.WORMHOLE:
        specific_url = BASE_URL_WORMHOLE_SPECIFIC
    elif CHIP_ARCH == ChipArchitecture.BLACKHOLE:
        specific_url = ""
    else:
        print(f"Unsupported CHIP_ARCH detected: {CHIP_ARCH}")
        sys.exit(1)

    # Download headers
    for header in HEADERS:
        header_url = f"{BASE_URL}/{header}"
        specific_header_url = f"{specific_url}/{header}" if specific_url else None

        try:
            print(f"Downloading {header} from {header_url}...")
            response = requests.get(header_url, timeout=10)
            response.raise_for_status()
            with open(os.path.join(HEADER_DIR, header), "wb") as f:
                f.write(response.content)
        except requests.RequestException:
            if specific_header_url:
                try:
                    print(f"Retrying {header} from {specific_header_url}...")
                    response = requests.get(specific_header_url, timeout=10)
                    response.raise_for_status()
                    with open(os.path.join(HEADER_DIR, header), "wb") as f:
                        f.write(response.content)
                except requests.RequestException:
                    print(f"Failed to download {header}")
                    sys.exit(1)
            else:
                print(f"Failed to download {header}")
                sys.exit(1)

    # Create the stamp file to indicate headers are downloaded
    with open(STAMP_FILE, "w") as f:
        f.write("Headers downloaded.\n")


def pytest_configure(config):
    log_file = "pytest_errors.log"
    # Clear the log file if it exists
    if os.path.exists(log_file):
        os.remove(log_file)
    logging.basicConfig(
        filename=log_file,
        level=logging.ERROR,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def pytest_runtest_logreport(report):
    # Capture errors when tests fail
    if report.failed:
        logging.error(f"Test {report.nodeid} failed: {report.longrepr}\n")


# Modify how the nodeid is generated
def pytest_collection_modifyitems(items):
    for item in items:
        # Modify the test item to hide the function name and only show parameters
        # item.nodeid is immutable, so we should modify how the test is represented
        if "::" in item.nodeid and "[" in item.nodeid:
            file_part, params_part = item.nodeid.split("::", 1)
            param_only = params_part.split("[", 1)[1]  # Extract parameters
            item._nodeid = f"{file_part}[{param_only}]"


def pytest_runtest_protocol(item, nextitem):
    """
    This hook can modify the test item before it's executed.
    We're going to set the test function name to an empty string.
    """
    # Modify the nodeid to show only the parameters, not the function name
    if "::" in item.nodeid and "[" in item.nodeid:
        _, param_part = item.nodeid.split("::", 1)
        param_only = param_part.split("[", 1)[1]  # Extract parameters
        item.name = f"[{param_only}]"

    # Continue the test execution as usual
    return None


def pytest_sessionstart(session):
    # Send ARC message for GO BUSY signal. This should increase device clock speed.
    ARC_COMMON_PREFIX = 0xAA00
    GO_BUSY_MESSAGE_CODE = 0x52
    arc_msg(
        device_id=0,
        msg_code=ARC_COMMON_PREFIX | GO_BUSY_MESSAGE_CODE,
        wait_for_done=True,
        arg0=0,
        arg1=0,
        timeout=10,
    )


def pytest_sessionfinish(session, exitstatus):
    BOLD = "\033[1m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"
    if _format_log:
        print(f"\n\n{BOLD}{YELLOW} Cases Where Dest Accumulation Turned On:{RESET}")
        for input_fmt, output_fmt in _format_log:
            print(f"{BOLD}{YELLOW}  {input_fmt} -> {output_fmt}{RESET}")

    # Send ARC message for GO IDLE signal. This should decrease device clock speed.
    ARC_COMMON_PREFIX = 0xAA00
    GO_IDLE_MESSAGE_CODE = 0x54
    arc_msg(
        device_id=0,
        msg_code=ARC_COMMON_PREFIX | GO_IDLE_MESSAGE_CODE,
        wait_for_done=True,
        arg0=0,
        arg1=0,
        timeout=10,
    )


# Skip decorators for specific architectures
# These decorators can be used to skip tests based on the architecture
# For example, if you want to skip a test for the "wormhole" architecture,
# decorate the test with @skip_for_wormhole.

skip_for_wormhole = pytest.mark.skipif(
    lambda: get_chip_architecture() == ChipArchitecture.WORMHOLE,
    reason="Test is not supported on Wormhole architecture",
)

skip_for_blackhole = pytest.mark.skipif(
    lambda: get_chip_architecture() == ChipArchitecture.BLACKHOLE,
    reason="Test is not supported on Blackhole architecture",
)
