import sys

def profile_issue_barrier(file_name):
    f = open(file_name, "r")
    lines = f.readlines()
    issue = []
    barrier = []
    for line in lines:
        lst = line.split()
        issue.append(int(lst[1]))
        barrier.append(int(lst[-1]))
    issue_avg = sum(issue)/len(issue)
    barrier_avg = sum(barrier)/len(barrier)
    print("issue: {:.2f} barrier: {:.2f}".format(issue_avg, barrier_avg))

file_name = sys.argv[1]
profile_issue_barrier(file_name)
