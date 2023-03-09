include gpai/csrc/module.mk

$(PYTHON_ENV)/lib/python3.8/site-packages/gpai.egg-link: python_env gpai/csrc/setup_inplace_link
	bash -c "source $(PYTHON_ENV)/bin/activate; cd gpai; pip install -e ."

gpai: gpai/csrc $(PYTHON_ENV)/lib/python3.8/site-packages/gpai.egg-link ;
