#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Install the SFPI release this tree pins (sfpi-version) into a directory.
#
#   install_sfpi.sh [--dest DIR]
#
# DIR becomes the toolchain root (compiler/, include/). The default is sfpi/ beside this script, where the
# test harness looks first; for any other directory, set LLK_SFPI_ROOT to it.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPILER="compiler/bin/riscv-tt-elf-g++"

TEMP_DIR=""
STAGING_DIR=""

cleanup() {
    [[ -n "$TEMP_DIR" && -d "$TEMP_DIR" ]] && rm -rf "$TEMP_DIR"
    [[ -n "$STAGING_DIR" && -d "$STAGING_DIR" ]] && rm -rf "$STAGING_DIR"
    return 0
}

usage() {
    echo "Usage: $0 [--dest DIR]"
    echo "  --dest DIR   Directory to install the toolchain into (default: $SCRIPT_DIR/sfpi)."
}

main() {
    local dest="$SCRIPT_DIR/sfpi"
    while [[ "$#" -gt 0 ]]; do
        case $1 in
            --dest)
                if [[ -z "${2:-}" ]]; then
                    echo "ERROR: --dest needs a directory" >&2
                    exit 1
                fi
                dest="$2"
                shift
                ;;
            --help) usage; exit 0 ;;
            *) echo "ERROR: Unknown option: $1" >&2; usage >&2; exit 1 ;;
        esac
        shift
    done

    # The destination is replaced as a whole, so it must be absent, empty, or an SFPI install already.
    if [[ -e "$dest" && ! -f "$dest/sfpi.version" && ! -f "$dest/$COMPILER" ]] &&
       [[ -n "$(ls -A "$dest" 2>/dev/null)" ]] ; then
        echo "ERROR: '$dest' exists and is not an SFPI install; refusing to replace it" >&2
        exit 1
    fi

    local version_file="$SCRIPT_DIR/sfpi-info.sh"
    if [[ ! -r "$version_file" ]]; then
        echo "ERROR: sfpi-info.sh not found or not readable at '$version_file'" >&2
        exit 1
    fi

    # shellcheck source=/dev/null
    eval local $($version_file SHELL txz)

    if [[ -f "$dest/sfpi.version" ]] &&
       [[ "$sfpi_version" == "$(cat "$dest/sfpi.version")" ]] ; then
        echo "SFPI is already at the correct version: $sfpi_version"
        exit 0
    fi

    if [[ -z $sfpi_hash ]] ; then
        echo "[ERROR] SFPI $sfpi_version $sfpi_pkg package for $sfpi_arch $sfpi_dist is not available" >&2
        exit 1
    fi

    trap cleanup EXIT

    echo "SFPI not present or out of date. Fetching version ${sfpi_version}..."
    TEMP_DIR=$(mktemp -d)
    if ! wget -q -P "$TEMP_DIR" --waitretry=5 --retry-connrefused "$sfpi_url/$sfpi_filename" ; then
        echo "ERROR: Failed to download $sfpi_url/$sfpi_filename" >&2
        exit 1
    fi
    if [ "$(${sfpi_hashtype}sum -b "${TEMP_DIR}/$sfpi_filename" | cut -d' ' -f1)" \
             != "$sfpi_hash" ] ; then
        echo "[ERROR] SFPI $sfpi_filename ${sfpi_hashtype} mismatch" >&2
        exit 1
    fi

    # Extract beside the destination and swap it in, so a failed extraction leaves the old install in place.
    echo "Extracting SFPI release..."
    if ! mkdir -p "$(dirname "$dest")" ||
       ! STAGING_DIR=$(mktemp -d "$(dirname "$dest")/.sfpi-install.XXXXXX") ; then
        echo "ERROR: Cannot write to '$(dirname "$dest")'" >&2
        exit 1
    fi
    # The archive holds one directory, sfpi/.
    if ! tar xJf "$TEMP_DIR/$sfpi_filename" -C "$STAGING_DIR" || [[ ! -f "$STAGING_DIR/sfpi/$COMPILER" ]]; then
        echo "ERROR: Failed to extract SFPI release from $sfpi_filename" >&2
        exit 1
    fi
    echo "$sfpi_version" > "$STAGING_DIR/sfpi/sfpi.version"
    rm -rf "$dest"
    if ! mv "$STAGING_DIR/sfpi" "$dest" ; then
        echo "ERROR: Failed to move the SFPI release to '$dest'" >&2
        exit 1
    fi

    echo "SFPI successfully installed version $sfpi_version to $dest"
    if [[ "$dest" != "$SCRIPT_DIR/sfpi" ]]; then
        echo "Set LLK_SFPI_ROOT=$dest for the test harness to use it."
    fi
}

main "$@"
