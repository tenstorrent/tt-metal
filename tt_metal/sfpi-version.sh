#! /bin/bash

# Compute SFPI release version information.
# Emit as evaluable shell assignments or CMAKE script
# This is the source of truth as to how we determine arch and distro names.
# Canonical location is in tenstorrent/tt-sfpi project's script directory.

# Define release, update this bit for new release
sfpi_version=7.5.0
sfpi_aarch64_linux_deb_md5=bc0adb8f67a75b0f226ed2cbdf396a40
sfpi_aarch64_linux_rpm_md5=7701729b1f2567764183aab320502558
sfpi_aarch64_linux_txz_md5=15c8147501ffbc0d6869a7481f650987
sfpi_x86_64_linux_deb_md5=ff911ffe11a0a4bb40c48d87376295b0
sfpi_x86_64_linux_rpm_md5=c4899b46201329a9811b8cda33bc411c
sfpi_x86_64_linux_txz_md5=4ca0388aa696c9bf297651931938eb6b

# One ring to rule them all,
# One ring to find them,
# One ring to bring them all
# and in the darkness bind them

# For the realm of mortals, using the toolchain
# eval local $($FILE SHELL [$pkg]) -- generate sfpi variables to install
# $FILE CMAKE [$pkg] -- emit cmake script to set variables

# For the realm of elves, releasing the toolchain
# eval $($FILE RELEASE $version) -- generate sfpi variables for release
# $FILE *.md5 -- emit variable initializations from md5 file(s)

set -e

if [[ ${#1} = 0 ]] ; then
    cat >&2 <<EOF
Usage:
$0 *.md5	 - generate release information
$0 RELEASE \$VER - generate release names
$0 SHELL [\$PKG] - shell use for PKG
$0 CMAKE [\$PKG] - CMAKE use for PKG
EOF
    exit 1
fi

if [[ ${1-} =~ '.md5'$ ]] ; then
    # convert md5 files into release variables, to insert above
    version=
    for file in "$@"
    do
	ver="${file##*/sfpi_}"
	ver="${ver%%_*}"
	if [[ $ver != $version ]] ; then
	    version=$ver
	    echo sfpi_version=$version
	fi
    done
   sed 's/^\([0-9a-f]*\) \*sfpi_[^_]*_\([^.]*\)\.\(.*\)$/sfpi_\2_\3_md5=\1/' "$@"
   exit 0
fi

if [[ ${1-} = RELEASE ]] ; then
    # releaser of sfpi-version
    sfpi_version=$2
fi

# define host system
sfpi_dist=unknown
sfpi_pkg=
if [[ -r /etc/os-release ]] ; then
    source /etc/os-release
    # See if ID_LIKE indicates a debian or fedora clone
    for like in $ID_LIKE
    do
	case $like in
	    debian) ID=debian; break;;
	    fedora) ID=fedora; break;;
	esac
    done

    if [[ ${1-} = RELEASE ]] ; then
	sfpi_releaser=$ID
    fi

    # debian and fedora are sufficiently close to treat as one, modulo
    # packaging system. We endeavor to build on a common denominator
    # system and translate package dependencies.
    case $ID in
	debian) sfpi_dist=linux sfpi_pkg=deb;;
	fedora) sfpi_dist=linux sfpi_pkg=rpm;;
	*) sfpi_dist=$ID;;
    esac
fi
sfpi_arch=$(uname -m)

# define download location & name
sfpi_repo=https://github.com/tenstorrent/sfpi
sfpi_url=$sfpi_repo/releases/download/$sfpi_version
sfpi_filename=sfpi_${sfpi_version}_${sfpi_arch}_${sfpi_dist}

if [[ ${1-} != RELEASE ]] ; then
    # querier of sfpi-version
    if [[ -n $2 ]] ; then
	sfpi_pkg=$2
    fi
    sfpi_filename+=".$sfpi_pkg"
    sfpi_md5=$(eval echo "\${sfpi_${sfpi_arch}_${sfpi_dist}_${sfpi_pkg}_md5-}")
    unset sfpi_builton
fi

# now emit definitions
for var in $(set -o posix ; set | sed -e '/^sfpi_/{s/=.*//;p}' -e d)
do
    eval val="\$$var"
    # relies on no inserted quoting for meta-characters
    if [[ ${1-} = CMAKE ]] ; then
	echo "set(SFPI${var#sfpi} \"$val\")"
    else
	echo "${var}='$val'"
    fi
done
