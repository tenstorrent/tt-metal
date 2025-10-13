# set SFPI release version information

sfpi_url=https://github.com/tenstorrent/sfpi/releases/download

# convert md5 file into these variables
# sed 's/^\([0-9a-f]*\) \*sfpi_\([-.0-9a-zA-Z]*\)_\([a-z0-9_A-Z]*\)\.\([a-z]*\)$/sfpi_version=\2'"\n"'sfpi_\3_\4_md5=\1/' *-build/sfpi_*.md5 | sort -u
sfpi_aarch64_deb_md5=bbe267282dc8c36554bc91354e47fc7d
sfpi_aarch64_rpm_md5=f146cc686ffd08f5d5bf55055f2010f2
sfpi_aarch64_txz_md5=d03da69379ea08386ce88dfb2bcdb88c
sfpi_version=7.3.0
sfpi_x86_64_deb_md5=100e9834fe6877a186440b241f69da65
sfpi_x86_64_rpm_md5=38def21cfa7cfebef8e9def61019f7cc
sfpi_x86_64_txz_md5=17970c8034b039f269d9a9db8e841e45
