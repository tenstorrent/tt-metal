// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

import {Octokit} from '@octokit/rest';
import {writeFileSync} from 'fs';

// Environment variables
const token       = process.env.GITHUB_TOKEN;
const currentRepo = process.env.GITHUB_REPOSITORY;

if (!token || !currentRepo)
{
    console.error('❌ Missing GITHUB_TOKEN or GITHUB_REPOSITORY.');
    process.exit(1);
}

const [defaultOwner, defaultRepoName] = currentRepo.split('/');
const octokit                         = new Octokit({auth: token});

// Constants for priority and overdue thresholds
const PRIORITY_RANK = {
    P0: 0,
    P1: 1,
    P2: 2,
    None: 3
};
const PRIORITY_DUE_DAYS = {
    P0: 20,
    P1: 30,
    P2: 60
};

// Helper functions
const getDaysOpen = (createdAt) => Math.floor((new Date() - new Date(createdAt)) / (1000 * 60 * 60 * 24));

const getPriority = (issue) => {
    const label = issue.labels.find((l) => typeof l.name === 'string' && /^P[0-2]$/.test(l.name));
    return label ? label.name : 'None';
};

const isOverdue = (priority, daysOpen) => daysOpen > (PRIORITY_DUE_DAYS[priority] || Infinity);

const getStyledLabels = (labels) => labels.map((label) => `<span class="label" style="background-color:#${label.color}">${label.name}</span>`).join(' ');

const countIssuesByPriority = (issues) => {
    const counts = {P0: 0, P1: 0, P2: 0, None: 0}; // Ensure all keys are initialized to 0
    issues.forEach((issue) => {
        const priority   = getPriority(issue);
        counts[priority] = (counts[priority] || 0) + 1;
    });
    return counts;
};

const fetchIssues = async ({owner, repo, filterLabel = null, tag, state = 'open'}) => {
    const allIssues = await octokit.paginate(octokit.rest.issues.listForRepo, {
        owner,
        repo,
        state,
        per_page: 100,
    });

    return allIssues
        .filter((issue) => !issue.pull_request) // Exclude pull requests
        .filter((issue) => !filterLabel || issue.labels.some((l) => typeof l.name === 'string' && l.name === filterLabel))
        .map((issue) => ({...issue, _sourceRepo: tag}));
};

const calculateClosedIssuesStats = (issues) => {
    const closingTimes = {P0: [], P1: [], P2: [], None: []};

    issues.forEach((issue) => {
        const priority = getPriority(issue);
        if (issue.closed_at)
        {
            const createdAt   = new Date(issue.created_at);
            const closedAt    = new Date(issue.closed_at);
            const daysToClose = Math.floor((closedAt - createdAt) / (1000 * 60 * 60 * 24));
            closingTimes[priority].push(daysToClose);
        }
    });

    const stats = {};
    for (const priority in closingTimes)
    {
        const times     = closingTimes[priority];
        const count     = times.length;
        const avg       = count > 0 ? (times.reduce((sum, time) => sum + time, 0) / count).toFixed(1) : 'N/A';
        stats[priority] = {count, avg};
    }

    return stats;
};

const formattedDate = new Intl.DateTimeFormat('en-GB', {day: '2-digit', month: 'long', year: 'numeric'}).format(new Date());

const countBounties = (openIssues, closedIssues) => {
    const hasBountyLabel = (issue) => issue.labels.some((label) => typeof label.name === 'string' && label.name.toLowerCase() === 'bounty');

    const openBounties   = openIssues.filter(hasBountyLabel).length;
    const closedBounties = closedIssues.filter(hasBountyLabel).length;

    return {open: openBounties, closed: closedBounties, total: openBounties + closedBounties};
};

const generateIssueRow = (issue) => {
    const priority     = getPriority(issue);
    const daysOpen     = getDaysOpen(issue.created_at);
    const createdAt    = new Intl.DateTimeFormat('en-US', {day: '2-digit', month: 'long', year: 'numeric'}).format(new Date(issue.created_at)); // Format date
    const labels       = getStyledLabels(issue.labels);
    const overdueClass = isOverdue(priority, daysOpen) ? 'overdue' : '';
    const reporter     = issue.user.login; // Reporter is the issue creator
    const assignees    = issue.assignees.length > 0 ? issue.assignees.map((assignee) => assignee.login).join(', ') : 'None';

    return `
  <tr class="${overdueClass}">
    <td>${issue._sourceRepo}</td>
    <td><a href="${issue.html_url}" target="_blank">#${issue.number}</a></td>
    <td><a href="${issue.html_url}" target="_blank">${issue.title}</a></td>
    <td>${priority}</td>
    <td>${createdAt}</td>
    <td>${daysOpen}</td>
    <td>${labels}</td>
    <td>${reporter}</td> <!-- New Reporter column -->
    <td>${assignees}</td>
  </tr>
`;
};

const generateHTMLReport = (rows, priorityCounts, closedIssuesStats, bountyCounts) => `
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>LLK Open Issues Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
body {
  font-family: "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  background-color: #f4f6f9;
  color: #333;
  padding: 20px;
  line-height: 1.4;
  font-size: 14px;
}
h1 {
  text-align: center;
  color: #444;
  margin-bottom: 20px;
  font-size: 20px;
}
.report-container {
  display: flex;
  justify-content: space-between;
  gap: 10px; /* Reduced gap between the two divs */
  margin-top: 10px;
  margin-bottom: 10px; /* Reduced gap between the divs and the table below */
}
.stats-container {
  flex: 2; /* Take 2/3 of the width */
  background: #fff;
  border: 1px solid #ddd; /* Added border for better contrast */
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  padding: 15px;
  display: flex;
  flex-direction: column;
  justify-content: space-between; /* Ensure content is spaced evenly */
}
.chart-container {
  flex: 1; /* Take 1/3 of the width */
  background: #fff;
  border: 1px solid #ddd; /* Added border for better contrast */
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  padding: 15px;
  display: flex;
  flex-direction: column;
  justify-content: center; /* Center the chart */
  height: 50%; /* Reduced height by 10% */
}
.chart-container canvas {
  height: 100%; /* Ensure the canvas fills the container */
  max-height: 100%; /* Prevent overflow */
}
table {
  border-collapse: collapse;
  width: 100%;
  margin-top: 10px; /* Reduced margin */
  background: #fff;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  font-size: 14px; /* Increased font size for better readability */
}
th, td {
  padding: 10px; /* Slightly increased padding */
  border: 1px solid #ddd;
  text-align: left;
  vertical-align: top;
}
td {
  font-size: 16px; /* Increased font size for table entries */
}
td a {
  font-size: 16px; /* Match link font size to table entries */
  color: #007bff;
  text-decoration: none;
}
td a:hover {
  text-decoration: underline;
}
th {
  background-color: #007bff;
  color: #fff;
  font-weight: bold;
}
tr:nth-child(even) {
  background-color: #f9f9f9;
}
tr:hover {
  background-color: #f1f5ff;
}
tr.overdue {
  background-color: #ffe6e6 !important;
}
.label {
  display: inline-block;
  padding: 2px 6px;
  margin: 1px 3px;
  font-size: 12px; /* Increased font size for labels */
  font-weight: 500;
  color: #fff;
  border-radius: 12px;
}
.label:hover {
  opacity: 0.9;
}
footer {
  text-align: center;
  margin-top: 20px;
  font-size: 12px;
  color: #666;
}
</style>
</head>
<body>
<h1>LLK Open Issues Report — ${formattedDate}</h1>
<div class="report-container">
<div class="stats-container">
  <h2>Closed Issues Statistics</h2>
  <table>
    <thead>
      <tr>
        <th>Priority</th>
        <th>Closed Count</th>
        <th>Avg. Closing Time (Days)</th>
      </tr>
    </thead>
    <tbody>
      <tr><td>P0</td><td>${closedIssuesStats.P0.count}</td><td>${closedIssuesStats.P0.avg}</td></tr>
      <tr><td>P1</td><td>${closedIssuesStats.P1.count}</td><td>${closedIssuesStats.P1.avg}</td></tr>
      <tr><td>P2</td><td>${closedIssuesStats.P2.count}</td><td>${closedIssuesStats.P2.avg}</td></tr>
      <tr><td>None</td><td>${closedIssuesStats.None.count}</td><td>${closedIssuesStats.None.avg}</td></tr>
    </tbody>
  </table>
  <div style="margin-top: 20px; font-size: 1.1em;">
    <strong>Bounties Summary:</strong><br>
    Open Bounties: ${bountyCounts.open} <br>
    Closed Bounties: ${bountyCounts.closed}
  </div>
</div>
<div class="chart-container">
  <h2>Open Issues Distribution</h2>
  <canvas id="priorityChart"></canvas>
</div>
</div>
<table>
<thead>
  <tr>
    <th>Repository</th>
    <th>Issue</th>
    <th>Title</th>
    <th>Priority</th>
    <th>Created At</th>
    <th>Days Open</th>
    <th>Labels</th>
    <th>Reporter</th>
    <th>Assignees</th>
  </tr>
</thead>
<tbody>
  ${rows}
</tbody>
</table>
<footer>
  Generated by LLK Issue Tracker © 2025
</footer>
<script>
const ctx = document.getElementById('priorityChart').getContext('2d');
new Chart(ctx, {
  type: 'pie',
  data: {
    labels: ['P0', 'P1', 'P2', 'None'],
    datasets: [{
      data: [${priorityCounts.P0}, ${priorityCounts.P1}, ${priorityCounts.P2}, ${priorityCounts.None}],
      backgroundColor: ['#ff6384', '#36a2eb', '#ffce56', '#cccccc'],
    }]
  },
  options: {
    responsive: true,
    plugins: {
      legend: {
        position: 'bottom',
      },
      tooltip: {
        callbacks: {
          label: function (context) {
            const value = context.raw;
            const total = context.dataset.data.reduce((sum, val) => sum + val, 0);
            const percentage = ((value / total) * 100).toFixed(1) + '%';
            return context.label + ': ' + value + ' (' + percentage + ')';
          }
        }
      }
    }
  }
});
</script>
</body>
</html>
`;

(async () => {
    try
    {
        // Fetch open issues
        const localIssues = await fetchIssues({
            owner: defaultOwner,
            repo: defaultRepoName,
            tag: 'tt-llk',
        });

        const ttMetalIssues = await fetchIssues({
            owner: 'tenstorrent',
            repo: 'tt-metal',
            filterLabel: 'LLK',
            tag: 'tt-metal',
        });

        // Fetch closed issues
        const localClosedIssues = await fetchIssues({
            owner: defaultOwner,
            repo: defaultRepoName,
            tag: 'tt-llk',
            state: 'closed',
        });

        const ttMetalClosedIssues = await fetchIssues({
            owner: 'tenstorrent',
            repo: 'tt-metal',
            filterLabel: 'LLK',
            tag: 'tt-metal',
            state: 'closed',
        });

        // Combine and sort open issues
        const combinedIssues = [...localIssues, ...ttMetalIssues];
        const sortedIssues   = combinedIssues.sort((a, b) => {
            const aPriority    = getPriority(a);
            const bPriority    = getPriority(b);
            const priorityDiff = PRIORITY_RANK[aPriority] - PRIORITY_RANK[bPriority];
            return priorityDiff !== 0 ? priorityDiff : new Date(a.created_at) - new Date(b.created_at);
        });

        // Count issues by priority
        const priorityCounts = countIssuesByPriority(combinedIssues);

        // Calculate average closing times
        const closedIssues      = [...localClosedIssues, ...ttMetalClosedIssues];
        const closedIssuesStats = calculateClosedIssuesStats(closedIssues);
        const bountyCounts      = countBounties(combinedIssues, closedIssues);

        // Generate HTML report
        const rows = sortedIssues.map(generateIssueRow).join('');
        const html = generateHTMLReport(rows, priorityCounts, closedIssuesStats, bountyCounts);

        // Write to file
        writeFileSync('sorted-issues.html', html);
        console.log('✅ HTML report generated: sorted-issues.html');
    }
    catch (err)
    {
        console.error('❌ Failed to generate report:', err);
        process.exit(1);
    }
})();
