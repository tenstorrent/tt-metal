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

# download root location
sfpi_repo=https://github.com/tenstorrent/sfpi
sfpi_url=$sfpi_repo/releases/download

# define release
sfpi_version=7.3.0-naming-30304
sfpi_x86_64_linux_deb_md5=fd4feae9df6e6ddfea12e77534ff6b3a
sfpi_x86_64_linux_rpm_md5=96a613cd8cdb01a2ac3cf55c63ee56c7
sfpi_x86_64_linux_txz_md5=7bbbc772debaaf34f5801e429ded6a0b

case "${1-}" in
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
