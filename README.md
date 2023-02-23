# Getting up and running

1. With ``doxygen`` installed, clone this repo and build the docs.
```
git clone git@<HOST>:<REPO>.git --recurse-submodules
cd <REPO>
doxygen
cd docs
python3 -m venv env
source env/bin/activate
pip install -r requirements-dev.txt
PORT=<port> make all
```
This will build HTML pages for the docs and launch a web server on the port `<port>`.

2. Navigate to the docs page.

Navigate your web browser to `http://<ip address>:<port>`, where `<ip address>` is the IP address of the machine on which you launched the web server. For example: `http://10.250.37.37:4242`

3. Follow the instructions on the docs page. :)
