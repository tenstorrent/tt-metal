#! /bin/bash

# Set SFPI release version information.
# This is the source of truth as to how we determine arch and distro names.
# Canonical location is in tenstorrent/tt-sfpi project's script directory.

# source $FILE -- generate sfpi variables for use by shell
# $FILE CMAKE -- emit cmake script to set variables
# $FILE *.md5 -- emit variable initializations from md5 file(s)

case "${1-}" in
    *.md5)
	# convert md5 files into release variables
	sed 's/^\([0-9a-f]*\) \*sfpi_\([-.0-9a-zA-Z]*\)_\([a-z0-9_A-Z]*\)\.\([a-z]*\)$/sfpi_version=\2'"\n"'sfpi_\3_\4_md5=\1/' "$@" \
	    | sort -u
	exit 0
	;;
esac

# define host system
sfpi_dist=unknown
if [[ -r /etc/os-release ]] ; then
    # some distros are sufficiently similar for common handling
    sfpi_dist=$(eval $(grep '^ID=' /etc/os-release) ; \
		case "$ID" in \
		    debian|ubuntu|redhat|centos|rhel|almalinux) ID=linux ;; \
		esac ; \
		echo "$ID")
fi
sfpi_arch=$(uname -m)

# define release
sfpi_version=7.3.0-ext-29186
sfpi_x86_64_linux_deb_md5=23c4547bf95fb2f4e148fb1da0e433a2
sfpi_x86_64_linux_rpm_md5=7112d6ed4885ddf4eaa40521de96b479
sfpi_x86_64_linux_txz_md5=4eff7968d9c2851793197a38b44178df

sfpi_repo=https://github.com/tenstorrent/sfpi
sfpi_filename=sfpi_${sfpi_version}_${sfpi_arch}_${sfpi_dist}

if ! [[ -z ${1-} ]] ; then
    # querier of sfpi-version
    sfpi_filename+=".$1"
    sfpi_url=$sfpi_repo/releases/download/$sfpi_version
    sfpi_md5=$(eval echo "\${sfpi_${sfpi_arch}_${sfpi_dist}_${1}_md5:-}")
fi

case "${2-}" in
    CMAKE)
	# emit as cmake script
	for var in $(set -o posix ; set | grep '^sfpi_')
	do
	    # relies on no inserted quoting for meta-characters
	    name=${var%%=*}
	    echo "set(SFPI${name#sfpi} \"${var#*=}\")"
	done
	;;
esac
