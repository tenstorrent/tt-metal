#!/bash/bin

# Visulize the debug website from local machine by executing this script on local machine
# ssh <VM> -L 127.0.0.1:8888:127.0.0.1:8050
# Open http://localhost:8888 on local machine
ssh ubuntu@172.27.44.88 -L 127.0.0.1:8888:127.0.0.1:8050
