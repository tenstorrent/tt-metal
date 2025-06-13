#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e  # Exit immediately if a command exits with a non-zero status
set -o pipefail  # Fail if any command in a pipeline fails

version_file=$(dirname $0)/sfpi-version.sh
if ! [[ -r $version_file ]] ; then
    echo "sfpi-version.sh not found" >&2
    exit 1
fi

eval $(grep -v '^#' $version_file)
sfpi_arch_os=$(uname -m)_$(uname -s)
sfpi_txz_md5=$(eval echo "\$sfpi_${sfpi_arch_os}_txz_md5")
if [ -z "$sfpi_txz_md5" ] ; then
    echo "SFPI tarball for ${sfpi_arch_os} is not available" >&2
    exit 1
fi
if [ ! -e "sfpi/sfpi.version" ] || [ $(cat "sfpi/sfpi.version") != "$sfpi_version" ] ; then
    echo "SFPI not present or out of date, fetching ${sfpi_version}"
    TEMP_DIR=$(mktemp -d)
    if ! wget -P $TEMP_DIR --waitretry=5 --retry-connrefused "$sfpi_url/$sfpi_version/sfpi-${sfpi_arch_os}.txz" ; then
        echo "ERROR: Failed to download $sfpi_url/$sfpi_version/sfpi-${sfpi_arch_os}.txz" >&2
        exit 1
    fi
    if [ $(md5sum -b "$TEMP_DIR/sfpi-${sfpi_arch_os}.txz" | cut -d' ' -f1) \
	     != "$sfpi_txz_md5" ] ; then
	echo "ERROR: SFPI sfpi-${sfpi_arch_os}.txz md5 mismatch" >&2
	rm -rf $TEMP_DIR
	exit 1
    fi
    rm -rf sfpi
    if ! tar -xJf $TEMP_DIR/sfpi-${sfpi_arch_os}.txz; then
        echo "ERROR: Failed to extract SFPI release" >&2
        exit 1
    fi
    rm -rf $TEMP_DIR
    echo "${sfpi_version}" > sfpi/sfpi.version
    echo "SFPI installed version $sfpi_version"
else
    echo "SFPI already correct version $sfpi_version"
fi
