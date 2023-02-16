include ll_buda_bindings/csrc/module.mk

$(PYTHON_ENV)/lib/python3.8/site-packages/ll_buda_bindings.egg-link: python_env ll_buda_bindings/csrc/setup_inplace_link
	bash -c "source $(PYTHON_ENV)/bin/activate; cd ll_buda_bindings; pip install -e ."

ll_buda_bindings: ll_buda_bindings/csrc $(PYTHON_ENV)/lib/python3.8/site-packages/ll_buda_bindings.egg-link ;
