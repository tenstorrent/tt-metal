# Getting up and running

We've decided to put all of the information for this repo into our official documentation,
including info to get off the ground and started. Please read and perform the
following to build and view the docs.

1. Clone this repo and navigate to its folder.
```
git clone git@<HOST>:<REPO>.git --recurse-submodules
cd <REPO>
```

2. We must install docs dependencies, such as the virtualenv and Doxygen.
```
source ./run_docs_setup.sh
```

3. Now activate the environment and build HTML pages for the docs and launch a
   web server on the port `<port>`.

```
cd docs
source env/bin/activate
PORT=<port> make all
```

4. Navigate to the docs page.

Navigate your web browser to `http://<ip address>:<port>`, where `<ip address>`
is the IP address of the machine on which you launched the web server. For
example: `http://10.250.37.37:4242`, for port ``4242``.

5. Open a new terminal window and follow the `Getting Started` instructions on the docs page. :)
