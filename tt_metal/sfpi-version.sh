# set SFPI release version information

sfpi_version=6.21.0-gcc-15
sfpi_url=https://github.com/tenstorrent/sfpi/releases/download

# convert md5 file into these variables
# sed 's/^\([0-9a-f]*\) \*sfpi_[-.0-9a-zA-Z]*_\([a-z0-9_A-Z]*\)\.\([a-z]*\)$/sfpi_\2_\3_md5=\1/'
sfpi_x86_64_txz_md5=d0e6c5ff6d9ecb4f420601485c5cdb50
sfpi_x86_64_deb_md5=896c4f2053c184145a07dc9cf2559a62
sfpi_x86_64_rpm_md5=bd288676ed67fb0969cd129346e06449
