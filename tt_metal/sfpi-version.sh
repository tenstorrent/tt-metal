#! /bin/bash

# Set SFPI release version information.
# This is the source of truth as to how we determine arch and distro names.
# Canonical location is in tenstorrent/tt-sfpi project's script directory.

# eval local $($FILE SHELL $pkgext) -- generate sfpi variables for shell consumer
# $FILE CMAKE $pkgext -- emit cmake script to set variables
# eval $($FILE RELEASE $version) -- generate sfpi variables for release
# $FILE *.md5 -- emit variable initializations from md5 file(s)

# define release, update this bit for new release
sfpi_version=7.3.0-naming-30304
sfpi_x86_64_linux_deb_md5=4b2963e582bdf554894d31d75ffaad92
sfpi_x86_64_linux_rpm_md5=777fce0a26815235de84c53d5bab7954
sfpi_x86_64_linux_txz_md5=65c604a723cd1552765ffd4b6d41f22b

if [[ ${#1} = 0 ]] ; then
    cat >&2 <<EOF
Usage:
$0 *.md5	- generate release information
$0 RELEASE \$VER - generate release names
$0 SHELL \$PKG	- shell use for PKG
$0 CMAKE \$PKG	- CMAKE use for PKG
EOF
    exit 1
fi

if [[ ${1-} =~ '.md5'$ ]] ; then
   # convert md5 files into release variables
   (for file in "$@"
    do
	tmp="${file##*/sfpi_}"
	echo sfpi_version=${tmp%%_*}
    done) | sort -u
   sed 's/^\([0-9a-f]*\) \*sfpi_[^_]*_\([^.]*\)\.\(.*\)$/sfpi_\2_\3_md5=\1/' "$@"
   exit 0
fi

if [[ ${1-} = RELEASE ]] ; then
    # releaser of sfpi-version
    sfpi_version=$2
fi

# define host system
sfpi_dist=unknown
if [[ -r /etc/os-release ]] ; then
    source /etc/os-release

    case "${ID-}" in \
	"") ;;
	debian|ubuntu|redhat|centos|rhel|almalinux)
	    # some distros are sufficiently similar for common handling
	    sfpi_dist=linux ;;
	*)
	    sfpi_dist=$ID ;;
    esac
fi
sfpi_arch=$(uname -m)

# define download location & name
sfpi_repo=https://github.com/tenstorrent/sfpi
sfpi_url=$sfpi_repo/releases/download/$sfpi_version
sfpi_filename=sfpi_${sfpi_version}_${sfpi_arch}_${sfpi_dist}

if [[ ${1-} != RELEASE ]] ; then
    # querier of sfpi-version
    sfpi_filename+=".$2"
    sfpi_md5=$(eval echo "\${sfpi_${sfpi_arch}_${sfpi_dist}_${2}_md5-}")
fi

# now emit definitions
for var in $(set -o posix ; set | grep '^sfpi_')
do
    # relies on no inserted quoting for meta-characters
    name=${var%%=*}
    if [[ ${1-} = CMAKE ]] ; then
	echo "set(SFPI${name#sfpi} \"${var#*=}\")"
    else
	echo "${name}=${var#*=}"
    fi
done
