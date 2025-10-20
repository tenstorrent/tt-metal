#! /bin/bash

# Compute SFPI release version information.
# Emit as evaluable shell assignments or CMAKE script
# This is the source of truth as to how we determine arch and distro names.
# Canonical location is in tenstorrent/tt-sfpi project's script directory.

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

version_file="sfpi-version"
if [[ ${1-} =~ '.md5'$ ]] ; then
    # convert md5 files into sfpi-version file
    version=
    exit_code=0
    echo '# sfpi version information' >$version_file
    echo 'sfpi_repo=https://github.com/tenstorrent/sfpi' >>$version_file
    for file in "$@"
    do
	ver="${file##*/sfpi_}"
	ver="${ver%%_*}"
	if [[ $ver != $version ]] ; then
	    if [[ -n $version ]] ; then
	       echo "ERROR: Multiple versions" >&2
	       exit_code=1
	    fi
	    version=$ver
	    echo sfpi_version=$version >>$version_file
	fi
    done
   sed 's/^\([0-9a-f]*\) \*sfpi_[^_]*_\([^.]*\)\.\(.*\)$/sfpi_\2_\3_md5=\1/' "$@" >>$version_file
   exit $exit_code
fi

if [[ ${1-} = RELEASE ]] ; then
    # releaser of sfpi
    sfpi_version=$2
else
    source $(dirname $0)/$version_file
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
