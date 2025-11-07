
	// TLS symbols needed in linker script (one cannot set the type there)

	.macro tls name, section
	.pushsection \section,"awT"
	.global \name
	\name:
	.type \name, @object
	.popsection
	.endm

	tls __tdata_start, .tdata.start
	tls __tdata_end, .tdata.end

	tls __tbss_start, .tbss.start
	tls __tbss_end, .tbss.end
