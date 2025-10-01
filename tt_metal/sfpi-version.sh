# set SFPI release version information

sfpi_url=https://github.com/tenstorrent/sfpi/releases/download

# convert md5 file into these variables
# sed 's/^\([0-9a-f]*\) \*sfpi_\([-.0-9a-zA-Z]*\)_\([a-z0-9_A-Z]*\)\.\([a-z]*\)$/sfpi_version=\2'"\n"'sfpi_\3_\4_md5=\1/' *-build/sfpi_*.md5 | sort -u
sfpi_version=7.3.0-ttinsn2
sfpi_x86_64_deb_md5=fb07a458d6871decbd078c3185b344de
sfpi_x86_64_rpm_md5=233fb49a0a858afb182b58e827c9ef48
sfpi_x86_64_txz_md5=d57fdc2d864318321cfea05c6b2d8acd
